"""
FastAPI приложение для генерации блог-постов и отправки в Telegram
Интегрирует OpenAI API, Currents API и Telegram Bot API
"""

import os
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import openai
import requests
from datetime import datetime
import uvicorn

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Blog Post Generator & Telegram Bot API",
    description="API для генерации блог-постов на основе актуальных новостей и отправки в Telegram",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Получение API ключей из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Валидация наличия обязательных API ключей
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY не установлен в переменных окружения")
    raise ValueError("OPENAI_API_KEY должен быть установлен в переменных окружения")

if not CURRENTS_API_KEY:
    logger.error("CURRENTS_API_KEY не установлен в переменных окружения")
    raise ValueError("CURRENTS_API_KEY должен быть установлен в переменных окружения")

# Настройка OpenAI
openai.api_key = OPENAI_API_KEY

# Модели данных Pydantic (добавляем Telegram модели)
class TopicRequest(BaseModel):
    """Модель запроса для генерации поста"""
    topic: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="Тема для генерации блог-поста"
    )
    language: Optional[str] = Field(
        "en",
        description="Язык для поиска новостей (по умолчанию: 'en')"
    )
    max_news_items: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Максимальное количество новостей для использования (по умолчанию: 5)"
    )

class TelegramPost(BaseModel):
    """Модель запроса для отправки в Telegram"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Текст сообщения для отправки в Telegram"
    )
    image_url: Optional[str] = Field(
        None,
        description="URL изображения для отправки (опционально)"
    )
    parse_mode: Optional[str] = Field(
        "HTML",
        description="Режим разметки: 'HTML' или 'Markdown'"
    )

class NewsArticle(BaseModel):
    """Модель новостной статьи"""
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    published: Optional[str] = None

class GeneratedPost(BaseModel):
    """Модель сгенерированного поста"""
    topic: str
    title: str
    meta_description: str
    post_content: str
    news_used: List[NewsArticle]
    generated_at: str
    model_used: str

class HealthCheck(BaseModel):
    """Модель ответа проверки здоровья"""
    status: str
    timestamp: str
    services: Dict[str, str]

class TelegramSendResponse(BaseModel):
    """Модель ответа отправки в Telegram"""
    status: str
    message: Optional[str] = None
    telegram_response: Optional[Dict] = None
    error: Optional[str] = None

# ================== ФУНКЦИИ ДЛЯ TELEGRAM ==================

def send_to_telegram_bot(
    message: str, 
    image_url: Optional[str] = None,
    parse_mode: str = "HTML"
) -> Dict:
    """
    Отправляет сообщение в Telegram чат/канал
    
    Args:
        message: Текст сообщения
        image_url: URL изображения (опционально)
        parse_mode: Режим разметки ('HTML' или 'Markdown')
        
    Returns:
        Ответ Telegram API
        
    Raises:
        HTTPException: Если произошла ошибка при отправке
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Telegram credentials not configured"
        )
    
    try:
        # Если есть изображение
        if image_url:
            photo_data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "photo": image_url,
                "caption": message[:1024] if message else "",  # ограничение Telegram
                "parse_mode": parse_mode
            }
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                json=photo_data,
                timeout=10
            )
        # Если только текст
        else:
            text_data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": parse_mode
            }
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json=text_data,
                timeout=10
            )
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        logger.error("Таймаут при отправке в Telegram")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Таймаут при отправке сообщения в Telegram"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при отправке в Telegram: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ошибка Telegram API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при отправке в Telegram: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка отправки в Telegram: {str(e)}"
        )

# ================== СУЩЕСТВУЮЩИЕ ФУНКЦИИ ==================
# (get_recent_news, generate_content - остаются без изменений)

def get_recent_news(
    topic: str, 
    language: str = "en", 
    max_items: int = 5
) -> List[NewsArticle]:
    """
    Получает последние новости по заданной теме через Currents API
    """
    url = "https://api.currentsapi.services/v1/latest-news"
    
    params = {
        "language": language,
        "keywords": topic,
        "apiKey": CURRENTS_API_KEY
    }
    
    logger.info(f"Запрос новостей по теме: {topic}, язык: {language}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        news_data = response.json()
        
        if "news" not in news_data or not news_data["news"]:
            logger.warning(f"Новости по теме '{topic}' не найдены")
            return []
        
        articles = []
        for article in news_data["news"][:max_items]:
            news_article = NewsArticle(
                title=article.get("title", "Без заголовка"),
                description=article.get("description"),
                url=article.get("url"),
                published=article.get("published")
            )
            articles.append(news_article)
        
        logger.info(f"Получено {len(articles)} новостей по теме: {topic}")
        return articles
        
    except requests.exceptions.Timeout:
        logger.error(f"Таймаут при запросе новостей по теме: {topic}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Таймаут при запросе к Currents API"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе новостей: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ошибка при получении новостей: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке новостей: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки новостей: {str(e)}"
        )

def generate_content(topic: str, news_articles: List[NewsArticle]) -> Dict[str, str]:
    """
    Генерирует контент для блог-поста на основе темы и новостей
    """
    # Формирование контекста из новостей
    news_context = "\n".join(
        [f"- {article.title}: {article.description or 'Описание отсутствует'}" 
         for article in news_articles]
    ) if news_articles else "Актуальные новости по теме не найдены."
    
    try:
        # Генерация заголовка
        logger.info(f"Генерация заголовка для темы: {topic}")
        title_response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Создай привлекательный, точный и SEO-оптимизированный заголовок "
                    f"для статьи на тему '{topic}'. Учитывай следующий контекст новостей:\n"
                    f"{news_context}\n\n"
                    f"Заголовок должен быть кратким, интересным и отражать суть темы."
                )
            }],
            max_tokens=60,
            temperature=0.7,
            stop=["\n"]
        )
        title = title_response.choices[0].message.content.strip()
        
        # Генерация мета-описания
        logger.info(f"Генерация мета-описания для заголовка: {title}")
        meta_description_response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Напиши мета-описание для статьи с заголовком: '{title}'. "
                    f"Мета-описание должно быть:\n"
                    f"1. Информативным и содержательным (150-160 символов)\n"
                    f"2. Содержать основные ключевые слова\n"
                    f"3. Побуждать к прочтению\n"
                    f"4. Соответствовать контексту новостей:\n{news_context}"
                )
            }],
            max_tokens=120,
            temperature=0.5
        )
        meta_description = meta_description_response.choices[0].message.content.strip()
        
        # Генерация полного контента статьи
        logger.info(f"Генерация контента для темы: {topic}")
        post_content_response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Напиши подробную, хорошо структурированную статью на тему '{topic}'. "
                    f"Используй следующий контекст из актуальных новостей:\n{news_context}\n\n"
                    f"Требования к статье:\n"
                    f"1. Объем: не менее 1500 символов\n"
                    f"2. Структура: Введение, основная часть (3-4 раздела), заключение\n"
                    f"3. Каждый раздел должен иметь подзаголовок H2\n"
                    f"4. Включи анализ текущих трендов и примеры из новостей\n"
                    f"5. Используй маркированные списки где уместно\n"
                    f"6. Пиши в профессиональном, но доступном стиле\n"
                    f"7. Добавь практические рекомендации или выводы\n"
                    f"8. Убедись, что текст легко читается"
                )
            }],
            max_tokens=2000,
            temperature=0.7,
            presence_penalty=0.6,
            frequency_penalty=0.6
        )
        post_content = post_content_response.choices[0].message.content.strip()
        
        logger.info(f"Успешно сгенерирован контент для темы: {topic}")
        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }
        
    except openai.error.OpenAIError as e:
        logger.error(f"Ошибка OpenAI при генерации контента: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ошибка сервиса генерации контента: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при генерации контента: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации контента: {str(e)}"
        )

# ================== ЭНДПОИНТЫ ==================

@app.post(
    "/generate-post",
    response_model=GeneratedPost,
    status_code=status.HTTP_201_CREATED,
    summary="Сгенерировать блог-пост",
    description="Генерирует блог-пост на основе заданной темы с использованием актуальных новостей"
)
async def generate_post_api(request: TopicRequest):
    """
    Основной эндпоинт для генерации блог-поста
    """
    try:
        logger.info(f"Запрос на генерацию поста по теме: {request.topic}")
        
        # Получение новостей
        news_articles = get_recent_news(
            topic=request.topic,
            language=request.language,
            max_items=request.max_news_items
        )
        
        # Генерация контента
        content = generate_content(request.topic, news_articles)
        
        # Формирование ответа
        generated_post = GeneratedPost(
            topic=request.topic,
            title=content["title"],
            meta_description=content["meta_description"],
            post_content=content["post_content"],
            news_used=news_articles,
            generated_at=datetime.utcnow().isoformat(),
            model_used=OPENAI_MODEL
        )
        
        return generated_post
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Необработанная ошибка в generate-post: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@app.post(
    "/send-to-telegram",
    response_model=TelegramSendResponse,
    status_code=status.HTTP_200_OK,
    summary="Отправить сообщение в Telegram",
    description="Отправляет текстовое сообщение или сообщение с изображением в Telegram чат/канал"
)
async def send_to_telegram_api(post: TelegramPost):
    """
    Эндпоинт для отправки сообщений в Telegram
    """
    try:
        logger.info(f"Запрос на отправку в Telegram: {len(post.message)} символов")
        
        telegram_response = send_to_telegram_bot(
            message=post.message,
            image_url=post.image_url,
            parse_mode=post.parse_mode
        )
        
        return TelegramSendResponse(
            status="success",
            message="Сообщение успешно отправлено в Telegram",
            telegram_response=telegram_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка в send-to-telegram: {str(e)}")
        return TelegramSendResponse(
            status="error",
            error=str(e)
        )

@app.post(
    "/generate-and-send",
    status_code=status.HTTP_201_CREATED,
    summary="Сгенерировать и отправить в Telegram",
    description="Генерирует блог-пост и сразу отправляет его в Telegram"
)
async def generate_and_send_api(request: TopicRequest):
    """
    Комбинированный эндпоинт: генерирует пост и отправляет в Telegram
    """
    try:
        # Генерация поста
        generated_post = await generate_post_api(request)
        
        # Формирование сообщения для Telegram
        telegram_message = (
            f"<b>{generated_post.title}</b>\n\n"
            f"{generated_post.post_content[:1000]}...\n\n"
            f"📊 <i>Сгенерировано с использованием {generated_post.model_used}</i>"
        )
        
        # Отправка в Telegram
        telegram_response = send_to_telegram_bot(
            message=telegram_message,
            parse_mode="HTML"
        )
        
        return {
            "generated_post": generated_post,
            "telegram_sent": True,
            "telegram_response": telegram_response
        }
        
    except Exception as e:
        logger.error(f"Ошибка в generate-and-send: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации и отправке: {str(e)}"
        )

@app.get(
    "/health",
    response_model=HealthCheck,
    summary="Проверка здоровья сервиса",
    description="Проверяет статус работы всех компонентов сервиса"
)
async def health_check_api():
    """
    Эндпоинт для проверки работоспособности сервиса и внешних API
    """
    services_status = {}
    
    # Проверка OpenAI API
    try:
        test_openai = openai.Model.list(limit=1)
        services_status["openai"] = "healthy"
    except Exception as e:
        logger.warning(f"OpenAI API недоступен: {str(e)}")
        services_status["openai"] = "unhealthy"
    
    # Проверка Currents API
    try:
        test_currents = requests.get(
            "https://api.currentsapi.services/v1/available/regions",
            params={"apiKey": CURRENTS_API_KEY},
            timeout=5
        )
        if test_currents.status_code == 200:
            services_status["currentsapi"] = "healthy"
        else:
            services_status["currentsapi"] = "unhealthy"
    except Exception as e:
        logger.warning(f"Currents API недоступен: {str(e)}")
        services_status["currentsapi"] = "unhealthy"
    
    # Проверка Telegram Bot API
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            test_telegram = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
                timeout=5
            )
            if test_telegram.status_code == 200:
                services_status["telegram_bot"] = "healthy"
            else:
                services_status["telegram_bot"] = "unhealthy"
        else:
            services_status["telegram_bot"] = "not_configured"
    except Exception as e:
        logger.warning(f"Telegram Bot API недоступен: {str(e)}")
        services_status["telegram_bot"] = "unhealthy"
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values() 
        if status != "not_configured"
    ) else "degraded"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )

@app.get(
    "/",
    summary="Корневой эндпоинт",
    description="Информация о сервисе и доступные эндпоинты"
)
async def root_api():
    """
    Корневой эндпоинт с информацией о сервисе
    """
    return {
        "service": "Blog Post Generator & Telegram Bot API",
        "version": "1.1.0",
        "description": "Сервис для генерации блог-постов и отправки в Telegram",
        "endpoints": {
            "POST /generate-post": "Генерация блог-поста по теме",
            "POST /send-to-telegram": "Отправка сообщения в Telegram",
            "POST /generate-and-send": "Генерация поста и отправка в Telegram",
            "GET /health": "Проверка здоровья сервиса",
            "GET /docs": "Документация Swagger",
            "GET /redoc": "Альтернативная документация"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Глобальный обработчик исключений
    """
    logger.error(f"Необработанное исключение: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Внутренняя ошибка сервера",
            "error": str(exc)
        }
    )

# Точка входа для запуска приложения
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Запуск сервера на {host}:{port}")
    logger.info(f"Документация доступна по адресу: http://{host}:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
