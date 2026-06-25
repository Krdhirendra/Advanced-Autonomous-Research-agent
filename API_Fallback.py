# -------------- GOOGLE API KEY ROBUSTNESS & FALLBACK --------------- #
import os
import time
import asyncio
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI

def get_all_google_api_keys(env_file=".env"):
    """Extracts all GOOGLE_API_KEY values from the .env file to support fallback.
    Avoids the default behavior of python-dotenv where duplicate keys overwrite each other.
    """
    keys = []
    # Try different paths relative to the project
    paths_to_check = [
        env_file,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), env_file),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", env_file)
    ]
    
    resolved_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            resolved_path = p
            break
            
    if not resolved_path:
        return keys

    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if "GOOGLE_API_KEY" in line:
                parts = line.split("=", 1)
                if len(parts) == 2 and parts[0].strip() == "GOOGLE_API_KEY":
                    val = parts[1].strip().strip("'\"")
                    if val and val not in keys:
                        keys.append(val)
    return keys


class FallbackChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Custom ChatGoogleGenerativeAI subclass that implements automatic failover
    between multiple Google API keys with a 20-second cooldown period.
    """
    api_keys: list = []
    current_key_index: int = 0

    def __init__(self, api_keys: list, **kwargs):
        if not api_keys:
            raise ValueError("api_keys list cannot be empty")
        # Initialize the superclass with the first key
        super().__init__(google_api_key=api_keys[0], **kwargs)
        self.api_keys = api_keys
        self.current_key_index = 0

    def _switch_key(self):
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_index]

        print(f"\n[SECURITY ALERT] Google API Key at index {old_index} failed!")
        print(f"[SECURITY ALERT] Cooldown period of 20 seconds initiated...")
        time.sleep(20)
        print(f"[SECURITY ALERT] Switching to backup Google API Key at index {self.current_key_index}...")

        # Update key and client
        self.google_api_key = SecretStr(new_key)
        from google import genai
        self.client = genai.Client(api_key=new_key)

    async def _switch_key_async(self):
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_index]

        print(f"\n[SECURITY ALERT] Google API Key at index {old_index} failed!")
        print(f"[SECURITY ALERT] Cooldown period of 20 seconds initiated...")
        await asyncio.sleep(20)
        print(f"[SECURITY ALERT] Switching to backup Google API Key at index {self.current_key_index}...")

        # Update key and client
        self.google_api_key = SecretStr(new_key)
        from google import genai
        self.client = genai.Client(api_key=new_key)

    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as e:
            print(f"\n[LLM ERROR] Exception caught during invoke: {e}")
            if len(self.api_keys) > 1:
                self._switch_key()
                print("Retrying invocation...")
                return super().invoke(*args, **kwargs)
            else:
                raise e

    async def ainvoke(self, *args, **kwargs):
        try:
            return await super().ainvoke(*args, **kwargs)
        except Exception as e:
            print(f"\n[LLM ERROR] Async Exception caught during ainvoke: {e}")
            if len(self.api_keys) > 1:
                await self._switch_key_async()
                print("Retrying async invocation...")
                return await super().ainvoke(*args, **kwargs)
            else:
                raise e



#############################INVOKE##########################

# Initialize LLM
try:
    print(">Initializing the LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0
    )
    google_api_keys = get_all_google_api_keys(".env")
    if google_api_keys:
        print(f"✓ Found {len(google_api_keys)} Google API keys in .env. Initializing robust fallback LLM...")
        llm = FallbackChatGoogleGenerativeAI(
            api_keys=google_api_keys,
            model="gemini-3.1-flash-lite",
            temperature=0
        )
    else:
        print("⚠ No Google API keys found in .env. Falling back to default initialization...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0
        )
    print("✓ LLM initialized")
except Exception as e:
    print(f"✗ LLM initialization failed: {e}")
