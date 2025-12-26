"""
FastAPI приложение для генерации блог-постов на основе актуальных новостей
Интегрирует OpenAI API и Currents API для создания контента
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
    title="Blog Post Generator API",
    description="API для генерации блог-постов на основе актуальных новостей",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Получение API ключей из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")
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

# Модели данных Pydantic
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

# Функция для получения актуальных новостей через Currents API
def get_recent_news(
    topic: str, 
    language: str = "en", 
    max_items: int = 5
) -> List[NewsArticle]:
    """
    Получает последние новости по заданной теме через Currents API
    
    Args:
        topic: Тема для поиска новостей
        language: Язык новостей (по умолчанию: 'en')
        max_items: Максимальное количество новостей
        
    Returns:
        Список новостных статей
        
    Raises:
        HTTPException: Если произошла ошибка при получении новостей
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

# Функция для генерации контента с использованием OpenAI
def generate_content(topic: str, news_articles: List[NewsArticle]) -> Dict[str, str]:
    """
    Генерирует контент для блог-поста на основе темы и новостей
    
    Args:
        topic: Тема поста
        news_articles: Список новостных статей для контекста
        
    Returns:
        Словарь с заголовком, мета-описанием и содержимым поста
        
    Raises:
        HTTPException: Если произошла ошибка при генерации контента
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

# Эндпоинты API
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
    
    Args:
        request: Объект запроса с темой и параметрами
        
    Returns:
        Сгенерированный пост с метаданными
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
        # Пробрасываем уже обработанные HTTP исключения
        raise
    except Exception as e:
        logger.error(f"Необработанная ошибка в generate-post: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@app.get(
    "/health",
    response_model=HealthCheck,
    summary="Проверка здоровья сервиса",
    description="Проверяет статус работы всех компонентов сервиса"
)
async def health_check():
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
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values()
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
async def root():
    """
    Корневой эндпоинт с информацией о сервисе
    """
    return {
        "service": "Blog Post Generator API",
        "version": "1.0.0",
        "description": "Сервис для генерации блог-постов на основе актуальных новостей",
        "endpoints": {
            "POST /generate-post": "Генерация блог-поста по теме",
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
    # Получение параметров запуска из переменных окружения
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Запуск сервера на {host}:{port}")
    logger.info(f"Документация доступна по адресу: http://{host}:{port}/docs")
    
    # Запуск сервера
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
