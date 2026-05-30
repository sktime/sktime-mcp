import asyncio
from fastapi import FastAPI
from mcp.server.sse import SseServerTransport
from starlette.requests import Request

app = FastAPI()
sse = SseServerTransport("/messages")

async def handle_sse(scope, receive, send):
    async with sse.connect_sse(scope, receive, send) as streams:
        print("Connected via SSE")
        # Dummy server loop
        await asyncio.sleep(10)

app.add_route("/sse", handle_sse, methods=["GET"])
app.add_route("/messages", sse.handle_post_message, methods=["POST"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
