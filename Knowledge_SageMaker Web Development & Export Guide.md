# SageMaker Web Development & Export Guide

## 1. Environment & Setup
- **Work Directory**: `~/AI-Studio-ClearML`
- **Bash Tip**: When writing HTML via `echo`, always use **single quotes** (`'`) to wrap strings containing `!`. Using double quotes triggers a "bash: !DOCTYPE: event not found" error due to history expansion.

## 2. Creating a Web Game via Command Line
Run this to create a simple "Catch the Green Square" game:
\`\`\`bash
echo '<!DOCTYPE html><html><head><title>Terminal Game</title><style>body{text-align:center;font-family:sans-serif;background:#222;color:white}#game{width:400px;height:400px;border:2px solid #555;margin:20px auto;position:relative;background:#000;overflow:hidden}#player{width:30px;height:30px;background:#0f0;position:absolute;cursor:pointer;border-radius:4px}</style></head><body><h1>Catch the Green Square!</h1><p>Score: <span id="score">0</span></p><div id="game"><div id="player"></div></div><script>let score=0;const p=document.getElementById("player");const s=document.getElementById("score");function move(){const x=Math.floor(Math.random()*370);const y=Math.floor(Math.random()*370);p.style.left=x+"px";p.style.top=y+"px"}p.addEventListener("mousedown",()=>{score++;s.innerText=score;move()});move();</script></body></html>' > game.html
\`\`\`

## 3. Hosting the Server
Start a Python server on port 8080:
\`\`\`bash
python3 -m http.server 8080
\`\`\`

## 4. Accessing from Another Computer
There are two ways to view the file from another machine:

### Option A: Private Access (Authenticated)
If you are logged into your AWS account on the other computer, use the SageMaker Proxy URL:
- **Base URL**: To find and use the URL to view your HTML file, follow these steps:

    1. Start the Space: You cannot access a URL until the instance is running. Click the Run space button (top left of your screenshot). Wait for the status to change from "Stopped" to Running. Once it's running, an Open JupyterLab button will appear. Click it.

    2. Get the "Base" URL:when JupyterLab opens, look at your browser's address bar. It will look something like this:
https://[unique-id].studio.[region].sagemaker.aws/jupyterlab/default/lab/...

    3. Construct the "Proxy" URL: SageMaker uses a built-in proxy to let you view web servers (like the one you're trying to run). To see your index.html from another tab or computer: In your SageMaker Terminal, start your server:
    
    The final URL should look like this:
    https://[unique-id].studio.[region].sagemaker.aws/jupyterlab/default/proxy/8080/index.html
    - - **Access URL**: Replace `/lab` with `/proxy/8080/game.html`

### Option B: Public Access (Unauthenticated)
To share with someone who doesn't have AWS access, use an SSH tunnel (localhost.run):
\`\`\`bash
ssh -R 80:localhost:8080 nokey@localhost.run
\`\`\`
Give the person the resulting `https://...lhr.life` link.

