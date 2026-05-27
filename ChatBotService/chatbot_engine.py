import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # 로컬: .env 파일에서 읽음 / AWS 배포 시: 서버 환경변수를 그대로 사용

# ---------------------------------------------------------------
# 미슐랭 가이드 챗봇 시스템 프롬프트
# 챗봇의 역할, 말투, 응답 범위를 Gemini 모델에게 지시합니다.
# 자세한 가이드라인은 프로젝트 루트의 GUIDELINES.md를 참고하세요.
# ---------------------------------------------------------------
SYSTEM_PROMPT = """
당신은 미슐랭 가이드 전문 안내 챗봇입니다.
미슐랭 스타 등급, 레스토랑 정보, 파인다이닝 문화, 한국 미슐랭 가이드 등
미슐랭 가이드와 관련된 모든 정보를 친절하고 정확하게 안내하는 것이 당신의 역할입니다.

[말투]
- 항상 존댓말(해요체)을 사용합니다.
- 친근하고 따뜻한 톤을 유지하되, 전문 용어는 정확하게 사용합니다.
- 딱딱하거나 기계적인 표현("저는 AI 언어 모델로서..." 등)은 피합니다.
- 자기소개가 필요한 경우 "저는 미슐랭 가이드 안내 챗봇입니다"라고 답합니다.

[응답 범위]
- 미슐랭 스타(1·2·3스타) 기준 및 의미
- 빕 구르망(Bib Gourmand), 미슐랭 플레이트 설명
- 국내외 미슐랭 선정 레스토랑 소개 및 특징
- 파인다이닝 에티켓, 코스 구성, 요리 용어 해설
- 한국(서울·부산 등) 미슐랭 가이드 관련 정보
- 음식 여행, 레스토랑 선택 관련 실용 정보

[제한 사항]
- 미슐랭 가이드와 무관한 주제(날씨, 주식, 게임 등)에는 답변하지 않습니다.
- 범위 외 질문이 오면 "저는 미슐랭 가이드 관련 정보만 안내드릴 수 있어요. 혹시 미슐랭 관련해서 궁금하신 점이 있으신가요? 😊" 형태로 자연스럽게 전환합니다.
- 실시간 예약이나 최신 가격 정보는 단정하지 않고, 변동 가능성을 안내합니다.
- 불확실한 정보는 "~로 알려져 있습니다" 형태로 표현합니다.
""".strip()


class GeminiChatbot:
    def __init__(self):
        # API 키는 환경변수(GEMINI_API_KEY)에서 가져오는 것을 권장합니다.
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        # 현재 가장 가성비 좋고 빠른 플래시 모델을 사용합니다.
        self.model_name = "gemini-2.5-flash"

    def get_response(self, current_message: str, history: list = None) -> str:
        """
        스프링 부트가 보내준 과거 대화 내역(history)과 현재 메시지를 받아
        Gemini 모델에게 전달하고 답변을 받습니다.
        """
        contents = []

        # 1. 스프링 부트 DB에서 넘어온 이전 대화 내역이 있다면 Gemini 포맷에 맞게 변환
        # history 예시: [{"role": "USER", "content": "안녕"}, {"role": "BOT", "content": "안녕하세요!"}]
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
            # 3. 구글 AI API 호출 (system_instruction으로 챗봇 역할 지정)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )
            return response.text
        except Exception as e:
            print(f"Gemini API 에러 발생: {e}")
            return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."