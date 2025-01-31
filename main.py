import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from together import Together
import os
import httpx


TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

client = Together(api_key=TOGETHER_API_KEY)

def get_correct_answer(query: str):
    """Определяет правильный ответ на вопрос с вариантами с помощью DeepSeek AI (через Together API)."""
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": "Ты помощник, который выбирает правильный ответ на вопросы об Университете ИТМО."},
                {"role": "user", "content": f"Вопрос: {query}\nВыбери правильный ответ из предложенных вариантов и укажи только его номер (цифру от 1 до 10)."}
            ]
        )

        # Проверяем, что ответ содержит `choices`
        if response and response.choices:
            answer_text = response.choices[0].message.content.strip().split()[-1]
            return int(answer_text)

    except Exception as e:
        print(f"Ошибка при запросе к DeepSeek API: {e}")
        return -1
    
def search_relevant_links(query: str):
    """Ищет релевантные ссылки и reasoning по запросу с помощью Google Search API."""
    params = {"key": GOOGLE_SEARCH_API_KEY, "cx": GOOGLE_SEARCH_CX, "q": query}
    try:
        response = httpx.get(GOOGLE_SEARCH_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        sources = []
        reasoning = "Результаты поиска указывают на следующие источники:"
        for item in data.get("items", [])[:3]:  # Берем 3 лучших результата
            title = item.get("title")
            link = item.get("link")
            sources.append(HttpUrl(link))
            reasoning += f"\n- {title}: {link}"
        return sources, reasoning
    except Exception as e:
        print(f"Ошибка при поиске Google Search API: {e}")
        return [], "Не удалось найти релевантную информацию."

# Initialize
app = FastAPI()
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
        answer = get_correct_answer(body.query)
        sources, reasoning = search_relevant_links(body.query)

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
