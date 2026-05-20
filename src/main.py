from base64 import b64encode
from collections import defaultdict
import functools

import yaml
from httpx import AsyncClient, HTTPStatusError, TimeoutException
from fastapi import FastAPI, Response, BackgroundTasks
from pydantic import BaseModel, Field
from openai import AsyncOpenAI as OpenAI

app = FastAPI(title="Агентный уровень LLM-переводчика")

OPENAI_API_URL = "http://localhost:11434/v1/"
BATCH_SIZE_BYTES = 32

TRANSPORT_LEVEL_ENDPOINT = "http://10.105.237.166:8090/send"


client = OpenAI(base_url=OPENAI_API_URL, api_key="")
buffers = defaultdict(bytes)


class TranslationTaskData(BaseModel):
    id: str = Field(description="Идентификатор перевода")
    original_text: str = Field(description="Исходный текст")
    to_lang: str = Field(description="Целевой язык перевода")


class TranslationFragmentData(BaseModel):
    id: str = Field(description="Идентификатор перевода")
    fragment: str = Field(description="Фрагмент перевода в Base64")
    number: int = Field(description="Номер текущего фрагмента перевода")
    end: bool = Field(False, description="Признак последнего фрагмента перевода")
    error: bool = Field(False, description="Признак ошибки при переводе")


async def send_translation_fragment(data: TranslationFragmentData) -> None:
    print(data.model_dump_json(indent=4))

    try:
        async with AsyncClient(timeout=5.0) as http_client:
            response = await http_client.post(
                TRANSPORT_LEVEL_ENDPOINT,
                json=data.model_dump_json(),
            )
            response.raise_for_status()
    except HTTPStatusError as e:
        print(f"Got error, status code {e.response.status_code}")
    except TimeoutException:
        print("Transport layer is not responding, timeout occurred")


async def translate_test(data: TranslationTaskData) -> None:
    try:
        translation_stream = await client.chat.completions.create(
            model="translategemma-4b-it",
            messages=[
                {
                    "role": "user",
                    "content": f"ru to {data.to_lang}: {data.original_text}",
                },
            ],
            stream=True,
        )

        i = 0
        async for translation in translation_stream:
            if translation.choices and translation.choices[0].delta.content:
                translation_text = translation.choices[0].delta.content
                if not buffers[data.id]:
                    buffers[data.id] += translation_text.lstrip().encode("utf-8")
                else:
                    buffers[data.id] += translation_text.encode("utf-8")

            if len(buffers[data.id]) >= BATCH_SIZE_BYTES:
                i += 1
                fragment = buffers[data.id][:BATCH_SIZE_BYTES]
                buffers[data.id] = buffers[data.id][BATCH_SIZE_BYTES:]

                await send_translation_fragment(
                    TranslationFragmentData(
                        id=data.id,
                        fragment=b64encode(fragment).decode("ascii"),
                        number=i,
                    )
                )

        if buffers[data.id]:
            await send_translation_fragment(
                TranslationFragmentData(
                    id=data.id,
                    fragment=b64encode(buffers[data.id]).decode("ascii"),
                    number=i + 1,
                    end=True,
                )
            )
    except HTTPStatusError:
        async with AsyncClient(timeout=5.0) as http_client:
            response = await http_client.post(
                TRANSPORT_LEVEL_ENDPOINT,
                json=TranslationFragmentData(
                    id=data.id, fragment="", number=0, error=True
                ).model_dump_json(),
            )
            try:
                response.raise_for_status()
            except HTTPStatusError as e:
                print(f"Got error, status code {e.response.status_code}")


@app.post(
    "/transfer",
    summary="Отправить задание по переводу текста",
    response_description="Задание по переводу принято в работу",
    tags=["Перевод текста"],
)
async def translate_text(
    data: TranslationTaskData, background_tasks: BackgroundTasks
) -> None:
    background_tasks.add_task(translate_test, data)


@app.get("/openapi.yaml", include_in_schema=False)
@functools.lru_cache()
def openapi_yaml() -> Response:
    return Response(
        yaml.dump(
            app.openapi(),
            sort_keys=False,
            allow_unicode=True,
        ),
        media_type="text/yaml",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
