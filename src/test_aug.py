from openai import OpenAI
import pandas as pd

# 1. Set up the local connection
llm_client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="ollama", # Required by the client, ignored by Ollama
)

# Make sure this matches the model you pulled (e.g., "llama3.1" or "qwen2.5")
LLM_MODEL = "dolphin-llama3" 

def _paraphrase_llm(text: str) -> str:
    print(f"Sending to {LLM_MODEL}...")
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Eres un experto compositor. Reescribe y parafrasea la siguiente letra "
                        "de canción en español. Cambia el vocabulario usando sinónimos y "
                        "altera la estructura, pero mantén el sentimiento original. "
                        "REGLA ESTRICTA: NO USES SALTOS DE LÍNEA. Todo el texto debe fluir en un "
                        "solo párrafo. Separa los versos con comas (,) y las estrofas con puntos (.). "
                        "No incluyas el título ni el artista."
                    )
                },
                # --- FEW-SHOT EXAMPLE: We SHOW the model exactly how to behave ---
                {
                    "role": "user",
                    "content": "title: Ejemplo, artist: Fake. Me duele el alma, cuando te vas, y me dejas solo. Vuelve pronto, te lo ruego, no me hagas sufrir."
                },
                {
                    "role": "assistant",
                    # Notice the output: totally different words, strictly one line, comma/period format!
                    "content": "Siento un gran vacío en mi interior, al verte partir, dejándome en total abandono. Regresa rápido a mi lado, te lo imploro, evita que siga padeciendo."
                },
                # --- ACTUAL INPUT ---
                {"role": "user", "content": text}
            ],
            temperature=0.7,  # Bumped to 0.7 for maximum synonym swapping
            max_tokens=2048
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # PYTHON FAILSAFE: Even if the model hallucinates line breaks, this forcefully removes them
        # and converts multiple spaces into a single space.
        clean_output = " ".join(raw_output.split())
        
        return clean_output
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # A sample lyric formatted with your exact comma and period rules
    sample_lyric = pd.read_csv("../data/task1/train_df.csv")["lyrics"].iloc[723]
    
    print("--- ORIGINAL LYRIC ---")
    print(sample_lyric)
    print("\n----------------------\n")
    
    result = _paraphrase_llm(sample_lyric)
    
    print("--- PARAPHRASED LYRIC ---")
    print(result)