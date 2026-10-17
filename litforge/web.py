"""Local demo; generation is serialized to bound CPU load and preserve model mode."""

import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .training import load_checkpoint


class GenerateRequest(BaseModel):
    prompt: str = Field(default="ROMEO:\n", min_length=1, max_length=512)
    max_new_tokens: int = Field(default=200, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=20, ge=1, le=256)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def create_app(checkpoint_path):
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path)
    lock = threading.Lock()
    app = FastAPI(
        title="LitForge",
        description="A small transformer trained from scratch; experimental text generation.",
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "parameters": model.parameter_count,
            "training_step": checkpoint["step"],
            "context_characters": model.config.context,
        }

    @app.post("/generate")
    def generate(request: GenerateRequest):
        try:
            tokens = torch.tensor([tokenizer.encode(request.prompt)], dtype=torch.long)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Return a clear busy response instead of accumulating unbounded work.
        if not lock.acquire(blocking=False):
            raise HTTPException(
                429, "The model is generating another passage. Try again shortly."
            )
        try:
            start = time.perf_counter()
            result = model.generate(
                tokens,
                request.max_new_tokens,
                request.temperature,
                request.top_k,
                request.seed,
            )
            text = tokenizer.decode(result[0].tolist())
            seconds = time.perf_counter() - start
            return {
                "text": text,
                "continuation": text[len(request.prompt) :],
                "generated_characters": request.max_new_tokens,
                "seconds": seconds,
            }
        finally:
            lock.release()

    return app


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LitForge — Writing Lab</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101218;color:#e9e7e1;font:17px system-ui,sans-serif}
main{max-width:920px;margin:70px auto;padding:24px}.eyebrow{color:#d8b77e;letter-spacing:3px;font-size:12px}
h1{font:64px Georgia,serif;margin:18px 0}p{color:#aaaebc;line-height:1.6}label{display:block;margin:20px 0 8px}
textarea,select{background:#191d27;color:inherit;border:1px solid #424859;border-radius:8px;padding:14px;font:inherit;width:100%}
textarea{min-height:110px;resize:vertical}.controls{display:flex;gap:20px}.controls>div{flex:1}input{width:100%}
button{margin:24px 0;padding:13px 26px;background:#d8b77e;border:0;border-radius:7px;font:600 16px system-ui;color:#17191d;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:21px/1.7 Georgia,serif;background:#191d27;border-radius:10px;padding:26px;min-height:160px}
#status{color:#aaaebc;font-size:14px}footer{margin-top:35px;color:#aaaebc;font-size:13px}@media(max-width:600px){main{margin:20px auto}h1{font-size:48px}.controls{flex-direction:column;gap:0}}
</style></head><body><main><div class="eyebrow">THE WRITING LAB</div><h1>LitForge</h1>
<p>Give a small, experimental language model an opening line. See how it continues in the style of Shakespeare.</p>
<form id="form"><label for="prompt">Your opening</label><textarea id="prompt" maxlength="512" required>ROMEO:
</textarea><div class="controls"><div><label for="temperature">Creativity: <span id="temp-value">0.8</span></label>
<input id="temperature" type="range" min="0.2" max="1.4" step="0.1" value="0.8"></div>
<div><label for="length">Passage length</label><select id="length"><option value="100">Short · 100 characters</option><option value="200" selected>Medium · 200 characters</option><option value="400">Long · 400 characters</option></select></div></div>
<button id="submit" type="submit">Continue the passage</button></form><div id="status" role="status" aria-live="polite"></div>
<pre id="output">Your passage will appear here.</pre><footer>Trained from scratch on public-domain text. Output may contain invented words or incoherent passages.</footer>
<script>
const form=document.getElementById('form'),button=document.getElementById('submit'),status=document.getElementById('status'),output=document.getElementById('output');
document.getElementById('temperature').oninput=e=>document.getElementById('temp-value').textContent=e.target.value;
form.onsubmit=async e=>{e.preventDefault();button.disabled=true;status.textContent='Writing…';try{
const response=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:document.getElementById('prompt').value,max_new_tokens:Number(document.getElementById('length').value),temperature:Number(document.getElementById('temperature').value)})});
const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Please check the prompt and settings.');
output.textContent=data.text;status.textContent=`${data.generated_characters} characters generated in ${data.seconds.toFixed(1)} seconds.`;
}catch(error){status.textContent=error.message;}finally{button.disabled=false;}};
</script></main></body></html>"""
