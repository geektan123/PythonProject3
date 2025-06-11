from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate-code")
async def generate_code(request: PromptRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash", contents=[f"Convert this Excel command into Office.js: {request.prompt}"]
        )
        return {"code": response.text if hasattr(response, "text") else "No response received"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)