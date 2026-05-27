import os
from google import genai
from google.genai import types

class GeminiChatbot:
    def __init__(self):
        # API 키는 환경변수에서 가져오거나 직접 입력합니다.
        # os.environ.get("GEMINI_API_KEY") 권장
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key="AIzaSyAk9DET1B8MXyH8ohrkL7jvI1Gw2JFjeJk")
        # 현재 가장 가성비 좋고 빠른 플래시 모델을 사용합니다.
        self.model_name = "gemini-2.5-flash"

    def get_response(self, current_message: str, history: list = None) -> str:
        """
        스프링 부트가 보내준 과거 대화 내역(history)과 현재 메시지를 받아
        Gemini 모델에게 전달하고 답변을 받습니다.
        """
        contents = []
        
        # 1. 스프링 부트 DB에서 넘어온 이전 대화 내역이 있다면 Gemini 포맷에 맞게 변환
        # history 예시: [{"sender": "USER", "text": "안녕"}, {"sender": "BOT", "text": "안녕하세요!"}]
        if history:
            for chat in history:
                role = "user" if chat["role"] == "USER" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=chat["content"])]
                    )
                )
        
        # 2. 이번에 유저가 보낸 새로운 질문 추가
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=current_message)]
            )
        )
        
        try:
            # 3. 구글 AI API 호출
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            return response.text
        except Exception as e:
            print(f"Gemini API 에러 발생: {e}")
            return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."