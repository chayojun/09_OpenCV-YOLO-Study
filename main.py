from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def text():
    return {"message": "안녕하세요"}


@app.post("/chat")

# 이 함수는 'Chatmsg' 틀에 맞는 데이터를 'msg'으로 받음
async def chat(msg: str):
    # 만약 사용자가 보낸 텍스트가 "안녕"과 같다면
    if msg == "안녕":
        # '안녕하겠어요'라는 메시지를 돌려줌
        return {"message": "안녕 하겠어요?"}
    # 만약 다른 텍스트를 보냈다면
    else:
        # 무슨 말씀인지 모르겠습니다'라는 메시지를 돌려줌
        return {"message": "뭔 소리여 이해할 수 있게 말해"}