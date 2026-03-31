# Knowledge Base: Prototyping APIs within SageMaker Studio

## 1. Concept: The "Sandbox" Analogy
In the following, we treat **SageMaker Studio** as a "Private Sandbox." 
- **The Sandbox**: A safe place to build and test your AI models.
- **The Limitation**: Like a real sandbox, what you build stays inside unless you build a "bridge" (Tunnel) to the outside world.

## 2. Use case: Technical Workflow (Internal Only)
To let a mobile app talk to a model running inside your Studio, follow these three layers:

### Layer A: The Model (Python Logic)
You write a script (e.g., `app.py`) using **FastAPI**. It acts as the "waiter" taking orders for your model.
\`\`\`python
from fastapi import FastAPI
app = FastAPI()

@app.get("/predict")
def predict(data: float):
    # Your model logic here
    return {"result": data * 2} 
\`\`\`

### Layer B: The Server (Local Execution)
Run the server in the Studio Terminal to occupy a local port:
\`\`\`bash
pip install fastapi uvicorn
uvicorn app:app --port 8080
\`\`\`

### Layer C: The Bridge (Public Tunnel)
Since Studio is private, we use an SSH tunnel to create a temporary public URL:
\`\`\`bash
ssh -R 80:localhost:8080 nokey@localhost.run
\`\`\`
*Teaching Note:* This provides a **temporary** link (e.g., `https://xyz.lhr.life`) for your mobile app to use.

## 3. Critical Classroom Takeaways
When explaining this to others, remember these three "Rules of the Studio":

| Rule | Explanation | Student Impact |
| :--- | :--- | :--- |
| **The "Session" Rule** | The API only lives as long as your Studio "Space" is running. | If you close your laptop, the App stops working. |
| **The "Security" Rule** | Studio-internal APIs are for **testing only**. They have no password protection. | Do not use real user data with this method. |
| **The "Proxy" Rule** | To see the API yourself, use the SageMaker URL with `/proxy/8080/`. | Great for checking if the server is up. |

## 4. Summary for Students
"We use SageMaker Studio to prove the concept. Once your mobile app successfully receives a 'Hello' from this Studio API, you have successfully crossed the hardest bridge in AI development: **Connectivity.**"

