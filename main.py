import os
import json
import requests
import logging
import datetime
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from requests.exceptions import RequestException

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY environment variable not set.")
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not TELEGRAM_TOKEN:
    logger.critical("TELEGRAM_TOKEN environment variable not set.")
    raise ValueError("TELEGRAM_TOKEN environment variable not set.")

DATA_SOURCES = {
    'sellers': "https://chainx-beta.vercel.app/api/seller",
    'warehouses': "https://chainx-beta.vercel.app/api/warehouse",
    'products': "https://chainx-beta.vercel.app/api/product",
    'logistics': "https://chainx-beta.vercel.app/api/logistic",
    'inspectors': "https://chainx-beta.vercel.app/api/inspector",
    'factories': "https://chainx-beta.vercel.app/api/factories"
}

FIELD_TYPES = {
    'products': {
        'product_name': 'string',
        'product_description': 'string',
        'batch_number': 'string',
        'factory_id': 'string',
        'product_price': 'number',
        'product_stock': 'number',
        'mrp': 'number',
    },
    'warehouses': {
        'name': 'string',
        'description': 'string',
        'latitude': 'number',
        'longitude': 'number',
        'contact_details': 'string',
        'warehouse_size': 'number',
        'balance': 'number',
        'factory_id': 'string',
        'product_id': 'string',
        'product_count': 'number',
        'logistic_count': 'number',
    },
    'sellers': {
        'name': 'string',
        'seller_id': 'string',
        'description': 'string',
        'latitude': 'number',
        'longitude': 'number',
        'contact_info': 'string',
        'balance': 'number',
        'products_count': 'number',
        'order_count': 'number',
    },
    'logistics': {
        'name': 'string',
        'logistic_id': 'string',
        'transportation_mode': 'string',
        'status': 'string',
        'contact_info': 'string',
        'shipment_cost': 'number',
        'product_id': 'string',
        'product_stock': 'number',
        'warehouse_id': 'string',
        'latitude': 'number',
        'longitude': 'number',
        'balance': 'number',
    },
    'inspectors': {
        'name': 'string',
        'inspector_id': 'string',
        'latitude': 'number',
        'longitude': 'number',
        'balance': 'number',
        'fee_charge_per_product': 'number',
    },
    'factories': {
        'name': 'string',
        'factory_id': 'string',
        'description': 'string',
        'latitude': 'number',
        'longitude': 'number',
        'contact_info': 'string',
        'product_count': 'number',
        'balance': 'number',
    },
}

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logger.critical(f"Failed to configure Gemini: {e}", exc_info=True)
    raise SystemExit(f"Fatal Error: Could not initialize Gemini AI: {e}")

async def detect_entity(query: str) -> str | None:
    prompt = f"""
    Analyze the following user query and identify which single entity type it is primarily about.
    Query: "{query}"
    Available entity types: {list(DATA_SOURCES.keys())}

    Respond ONLY with the single most relevant entity name from the list in lowercase.
    If no single entity seems clearly relevant, respond with "none".
    """
    try:
        response = await model.generate_content_async(prompt)
        entity = response.text.strip().lower() if hasattr(response, 'text') else "none"
        if entity == "none" or entity not in DATA_SOURCES:
            return None
        return entity
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API Error in detect_entity for query '{query}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in detect_entity for query '{query}': {e}", exc_info=True)
        raise

async def extract_filters(entity: str, query: str) -> dict:
    fields = ', '.join(FIELD_TYPES[entity].keys())
    prompt = f"""
    Analyze the following user query and extract the search filters for the specified entity.
    Entity: {entity}
    Available fields: {fields}
    Query: "{query}"

    For string fields, use "contains 'text'" for partial matches or "exact 'text'" for exact matches.
    For numerical fields, use operators like "<50", ">100", "=20", etc.
    If the query mentions a category (e.g., "electronics category"), interpret it as a filter on relevant fields like name or description if no explicit category field exists.
    Respond with a JSON object where each key is a field name, and the value is the condition string.
    If no specific filters are mentioned, respond with an empty object.
    Example response: {{"product_name": "contains 'apple'", "product_price": "<50"}}
    """
    try:
        response = await model.generate_content_async(prompt)
        response_text = response.text.strip()
        if response_text:
            try:
                filters = json.loads(response_text)
                if isinstance(filters, dict):
                    return filters
                else:
                    logger.warning(f"Gemini returned non-dict for filters: {response_text}")
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse filters JSON: {response_text}. Error: {e}")
                return {}
        else:
            return {}
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API Error in extract_filters for entity '{entity}' and query '{query}': {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in extract_filters: {e}", exc_info=True)
        return {}

def apply_filter(item: dict, filters: dict, entity: str) -> bool:
    for field, condition in filters.items():
        if field not in item:
            continue
        field_type = FIELD_TYPES.get(entity, {}).get(field)
        if field_type == 'string':
            if condition.startswith("contains '") and condition.endswith("'"):
                text = condition[len("contains '"):-1].lower()
                if text not in str(item[field]).lower():
                    return False
            elif condition.startswith("exact '") and condition.endswith("'"):
                text = condition[len("exact '"):-1].lower()
                if str(item[field]).lower() != text:
                    return False
            else:
                return False
        elif field_type == 'number':
            try:
                parts = condition.split(' ', 1)
                if len(parts) != 2:
                    return False
                op, value_str = parts
                value = float(value_str)
                item_value = float(item[field])
                if op == '<':
                    if item_value >= value:
                        return False
                elif op == '>':
                    if item_value <= value:
                        return False
                elif op == '=':
                    if item_value != value:
                        return False
                elif op == '<=':
                    if item_value > value:
                        return False
                elif op == '>=':
                    if item_value < value:
                        return False
                else:
                    return False
            except (ValueError, TypeError):
                return False
    return True

async def generate_conversational_response(query: str) -> str:
    prompt = f"""
    You are a friendly Telegram bot assistant named ChainX Bot. A user has sent you the message: "{query}".
    This message doesn't seem to request data about specific entities like sellers, products, or warehouses.
    Respond in a casual, friendly tone as if continuing a conversation. Keep the response short (1-2 sentences), appropriate, and engaging.
    If the query is a greeting like "hi" or "hello", greet them back and invite them to ask about data.
    Avoid generating responses that assume specific entity data unless explicitly mentioned.
    """
    try:
        response = await model.generate_content_async(prompt)
        response_text = response.text.strip() if hasattr(response, 'text') else "Hey! What's up?"
        return response_text
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API Error in generate_conversational_response for query '{query}': {e}")
        return "Oops, something went wrong. Want to try asking about some data?"
    except Exception as e:
        logger.error(f"Unexpected error in generate_conversational_response for query '{query}': {e}", exc_info=True)
        return "Hmm, I got a bit tangled up. How about asking for some seller or product info?"

async def fetch_data(entity: str) -> list | None:
    url = DATA_SOURCES.get(entity)
    if not url:
        return None
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=20))
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return None
    except RequestException as e:
        logger.error(f"HTTP request failed for {entity} at {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {url}. Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during data fetch for {entity}: {e}", exc_info=True)
        return None

def format_filters(filters: dict) -> str:
    if not filters:
        return ""
    conditions = []
    for field, condition in filters.items():
        conditions.append(f"{field} {condition}")
    return " where " + " and ".join(conditions)

def format_response(data: list, entity: str, filters: dict = None) -> str:
    if not data:
        if filters:
            return f"ℹ️ No {entity} found matching{format_filters(filters)}."
        else:
            return f"ℹ️ No {entity} found."
    formatters = {
        'factories': lambda d: (
            f"🏭 *{d.get('name', 'N/A')}* (ID: {d.get('factory_id', 'N/A')})\n"
            f"   Desc: _{d.get('description', 'N/A')}_\n"
            f"   📍 Location: Lat {d.get('latitude', 'N/A')}, Lon {d.get('longitude', 'N/A')}\n"
            f"   📞 Contact: {d.get('contact_info', 'N/A')}\n"
            f"   📦 Products Count: {d.get('product_count', 'N/A')}\n"
            f"   💰 Balance: {d.get('balance', 'N/A')}\n"
        ),
        'products': lambda d: (
            f"📦 *{d.get('product_name', d.get('name', 'N/A'))}* (ID: {d.get('product_id', 'N/A')})\n"
            f"   Desc: _{d.get('product_description', 'N/A')}_\n"
            f"   Batch: `{d.get('batch_number', 'N/A')}`\n"
            f"   🏭 Factory ID: {d.get('factory_id', 'N/A')}\n"
            f"   💲 Price: {d.get('product_price', d.get('price', 'N/A'))}\n"
            f"   Stock: {d.get('product_stock', d.get('stock', 'N/A'))}\n"
            f"   🏷️ MRP: {d.get('mrp', 'N/A')}\n"
            f"   ✅ Quality Checked: {'Yes' if d.get('quality_checked') else 'No'}\n"
            f"   🕵️ Inspection ID: {d.get('inspection_id', 'N/A')} (Fee Paid: {'Yes' if d.get('inspection_fee_paid') else 'No'})\n"
        ),
        'sellers': lambda d: (
            f"🏪 *{d.get('name', 'N/A')}* (ID: {d.get('seller_id', d.get('id', 'N/A'))})\n"
            f"   Desc: _{d.get('description', 'N/A')}_\n"
            f"   📍 Location: Lat {d.get('latitude', 'N/A')}, Lon {d.get('longitude', 'N/A')}\n"
            f"   📞 Contact: {d.get('contact_info', 'N/A')}\n"
            f"   💰 Balance: {d.get('balance', 'N/A')}\n"
            f"   📦 Products Count: {d.get('products_count', 'N/A')}\n"
            f"   Orders: {d.get('order_count', 'N/A')}\n"
        ),
        'warehouses': lambda d: (
            f"🏢 *{d.get('name', 'N/A')}* (ID: {d.get('warehouse_id', 'N/A')})\n"
            f"   Desc: _{d.get('description', 'N/A')}_\n"
            f"   📍 Location: Lat {d.get('latitude', 'N/A')}, Lon {d.get('longitude', 'N/A')}\n"
            f"   📞 Contact: {d.get('contact_details', 'N/A')}\n"
            f"   📐 Size: {d.get('warehouse_size', d.get('capacity', 'N/A'))} units\n"
            f"   💰 Balance: {d.get('balance', 'N/A')}\n"
            f"   🏭 Factory ID: {d.get('factory_id', 'N/A')}\n"
            f"   📦 Product ID: {d.get('product_id', 'N/A')} (Count: {d.get('product_count', 'N/A')})\n"
            f"   🚚 Logistics Count: {d.get('logistic_count', 'N/A')}\n"
        ),
        'logistics': lambda d: (
            f"🚚 *{d.get('name', f'Shipment {d.get('logistic_id', 'N/A')}')}* (ID: {d.get('logistic_id', 'N/A')})\n"
            f"   Mode: {d.get('transportation_mode', d.get('type', 'N/A'))}\n"
            f"   Status: {d.get('status', 'N/A')}\n"
            f"   📞 Contact: {d.get('contact_info', 'N/A')}\n"
            f"   💲 Cost: {d.get('shipment_cost', 'N/A')}\n"
            f"   📦 Product ID: {d.get('product_id', 'N/A')} (Stock: {d.get('product_stock', d.get('capacity', 'N/A'))})\n"
            f"   🏢 Warehouse ID: {d.get('warehouse_id', 'N/A')}\n"
            f"   📍 Current Loc: Lat {d.get('latitude', 'N/A')}, Lon {d.get('longitude', 'N/A')}\n"
            f"   Delivered: {'Yes' if d.get('delivered') else 'No'} (Confirmed: {'Yes' if d.get('delivery_confirmed') else 'No'})\n"
            f"   💰 Balance: {d.get('balance', 'N/A')}\n"
        ),
        'inspectors': lambda d: (
            f"🕵️ *{d.get('name', 'N/A')}* (ID: {d.get('inspector_id', 'N/A')})\n"
            f"   📍 Location: Lat {d.get('latitude', 'N/A')}, Lon {d.get('longitude', 'N/A')}\n"
            f"   💰 Balance: {d.get('balance', 'N/A')}\n"
            f"   💲 Fee/Product: {d.get('fee_charge_per_product', 'N/A')}\n"
        ),
    }
    default_formatter = lambda d: f"```json\n{json.dumps(d, indent=2)}\n```"
    formatter = formatters.get(entity, default_formatter)
    output_items = []
    for item in data:
        try:
            if isinstance(item, dict):
                output_items.append(formatter(item))
            else:
                logger.warning(f"Skipping non-dict item in data for entity {entity}: {item}")
        except Exception as e:
            item_id = "Unknown ID"
            if isinstance(item, dict):
                item_id = item.get(f'{entity[:-1]}_id', item.get('id', item.get('product_id', item.get('factory_id', 'Unknown ID'))))
            output_items.append(f"⚠️ Error formatting item: ID {item_id}")
    if filters:
        title = f"🔍 *Found {len(output_items)} {entity.capitalize()}{format_filters(filters)}*:\n\n"
    else:
        title = f"🔍 *Found {len(output_items)} {entity.capitalize()}*:\n\n"
    response_body = "\n\n---\n\n".join(output_items)
    return title + response_body

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = f"""
👋 Welcome to the *ChainX Data Bot*!

I can fetch and display information about various entities in the system or just chat a bit!

*Available Entities:*
{chr(10).join([f"- `{key.capitalize()}`" for key in DATA_SOURCES.keys()])}

*How to Use:*
Just tell me which entity you want to see data for, or say hi to chat!

*Examples:*
- `Show warehouses in New York`
- `List products with name containing apple`
- `Get inspectors with fee less than 50`
- `factories`
- `hi`

➡️ Type an entity name or ask for specific data!
    """
    if update.message:
        await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_query = update.message.text
    chat_id = update.message.chat_id
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    except Exception as e:
        logger.warning(f"Could not send typing action to chat {chat_id}: {e}")
    try:
        entity = await detect_entity(user_query)
        if not entity:
            conversational_response = await generate_conversational_response(user_query)
            await update.message.reply_text(conversational_response, parse_mode='Markdown')
            return
        all_data = await fetch_data(entity)
        if all_data is None:
            await update.message.reply_text(f"⚠️ Sorry, I couldn't fetch data for *{entity}*. The data source might be unavailable or returned an unexpected format. Please try again later.", parse_mode='Markdown')
            return
        filters = await extract_filters(entity, user_query)
        if filters:
            filtered_data = [item for item in all_data if apply_filter(item, filters, entity)]
        else:
            filtered_data = all_data
        response_text = format_response(filtered_data, entity, filters)
        MAX_LEN = 4096
        if len(response_text) > MAX_LEN:
            trunc_point = response_text.rfind('\n\n---\n\n', 0, MAX_LEN - 20)
            if trunc_point == -1:
                trunc_point = MAX_LEN - 20
            truncated_message = response_text[:trunc_point] + "\n\n[...]✂️"
            await update.message.reply_text(truncated_message, parse_mode='Markdown')
            await update.message.reply_text(f"ℹ️ The list for *{entity}* was too long and had to be shortened.", parse_mode='Markdown')
        else:
            await update.message.reply_text(response_text, parse_mode='Markdown')
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API Error processing query '{user_query}': {e}")
        await update.message.reply_text("⚠️ Sorry, I encountered an issue communicating with the AI analysis service. Please try again later.")
    except RequestException as e:
        logger.error(f"Data Fetching API Error for entity '{entity}' (query: '{user_query}'): {e}")
        entity_name = f" for *{entity}*" if entity else ""
        await update.message.reply_text(f"⚠️ Sorry, I couldn't reach the data source{entity_name}. Please check if the service is running or try again later.", parse_mode='Markdown')
    except ValueError as e:
        logger.error(f"Value Error processing query '{user_query}': {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Sorry, I encountered an issue processing the data or your query.")
    except Exception as e:
        logger.exception(f"Unhandled error processing message from chat {chat_id} (query: '{user_query}'): {e}")
        await update.message.reply_text("⚠️ An unexpected error occurred. I've logged the issue. Please try again or contact the administrator if the problem persists.")

def main():
    try:
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.run_polling()
    except Exception as e:
        logger.critical(f"Fatal error during bot setup or polling startup: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()