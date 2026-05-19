"""
SmartHandover - Simulador de Call Center (Apresentação Final)
Foco 100% no Pipeline Ensemble e Handover Automático.
Respostas de assistente ultrarrápidas com gTTS.
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gtts import gTTS
import tempfile

# A IMPORTAÇÃO QUE FALTAVA (O coração da vossa Tese)
from src.classifiers.pipeline_v4 import SmartHandoverPipelineV4

# --- CONFIGURAÇÃO DA VOZ (gTTS Seguro e Rápido) ---
def generate_bot_audio(text):
    """Gera voz rapidamente de forma assíncrona e segura"""
    tts = gTTS(text=text, lang='en', tld='com')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# --- CARREGAMENTO DO PIPELINE DA TESE ---
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("[Simulador] A carregar o pipeline V4 (Whisper + Wav2Vec2 + RoBERTa)...")
        pipeline = SmartHandoverPipelineV4()
    return pipeline

def get_instant_smart_reply(transcription):
    """
    Simula um LLM utilizando categorização avançada e fallbacks dinâmicos.
    """
    text = transcription.lower()
    import random
    
    # 1. Cumprimentos e Início de Chamada
    if any(word in text for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return "Hello! Thank you for calling. How can I help you today?"
        
    # 2. Problemas de Internet/Rede
    elif any(word in text for word in ["internet", "wifi", "connection", "router", "network", "slow", "down"]):
        return "It sounds like you're having connectivity issues. I can run a quick diagnostic on your network."
        
    # 3. Hardware e Dispositivos
    elif any(word in text for word in ["laptop", "computer", "phone", "screen", "battery", "broken", "turn on"]):
        return "I'm sorry your device is giving you trouble. Let's go through some quick troubleshooting steps."
        
    # 4. Faturação e Dinheiro
    elif any(word in text for word in ["bill", "pay", "charge", "invoice", "money", "expensive", "credit card"]):
        return "I can certainly help you review your billing details and explain any recent charges."
        
    # 5. Cancelamentos ou Devoluções
    elif any(word in text for word in ["cancel", "refund", "unsubscribe", "money back", "close account"]):
        return "I can help you with the cancellation process, but I would love to see if we can resolve the issue first."
        
    # 6. Gestão de Contas e Passwords
    elif any(word in text for word in ["password", "login", "account", "access", "email"]):
        return "I can assist you with resetting your credentials so you can access your account again."
        
    # 7. Respostas a frases curtas (Sim, Não, OK, etc.)
    elif len(text.split()) <= 3:
        short_replies = [
            "Could you elaborate a bit more on that?",
            "I'm listening. Please continue.",
            "Alright. What else can you tell me about the situation?"
        ]
        return random.choice(short_replies)
        
    # 8. O "CÉREBRO FALSO" (Fallbacks Genéricos)
    # Se o cliente disser algo completamente fora do guião, o bot usa uma destas respostas 
    # que se encaixam em 99% das frases humanas, simulando compreensão.
    else:
        fallbacks = [
            "I understand. Can you provide a few more details so I can find the best solution?",
            "Got it. Let me check my system for more information regarding that specific issue.",
            "Thank you for explaining. What specific outcome are you looking for today?",
            "I see. Let's work together to get this resolved for you as quickly as possible."
        ]
        return random.choice(fallbacks)

# --- LÓGICA DE NEGÓCIO E GRÁFICOS ---
def check_handover_logic(emotion_history):
    # Estilos CSS embutidos para os alertas
    style_ok = "background-color:#e8f5e9; color:#2e7d32; padding:15px; border-radius:8px; text-align:center; font-weight:bold; font-size:18px; border: 2px solid #66bb6a;"
    style_alert = "background-color:#ffebee; color:#c62828; padding:15px; border-radius:8px; text-align:center; font-weight:bold; font-size:18px; border: 2px solid #ef5350; animation: blinker 1s linear infinite;"
    
    if not emotion_history:
        return False, f"<div style='{style_ok}'>🟢 OPERADOR VIRTUAL (Ativo)</div>"
    
    latest = emotion_history[-1]
    soma_negativa = latest.get("frustration", 0) + latest.get("anger", 0)
    
    if soma_negativa >= 0.6:
        return True, f"<div style='{style_alert}'>🚨 HANDOVER ATIVADO: Pico Súbito! -> Transferindo...</div>"
    
    if len(emotion_history) == 3:
        avg_frust = sum(turn.get("frustration", 0) for turn in emotion_history) / 3.0
        avg_anger = sum(turn.get("anger", 0) for turn in emotion_history) / 3.0
        
        if (avg_frust + avg_anger) >= 0.45:
            return True, f"<div style='{style_alert}'>🚨 HANDOVER ATIVADO: Tendência Negativa! -> Transferindo...</div>"
            
    return False, f"<div style='{style_ok}'>🟢 OPERADOR VIRTUAL (Ativo)</div>"

def plot_emotion_trend(emotion_history):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    if not emotion_history:
        ax.text(0.5, 0.5, 'A aguardar chamada...', horizontalalignment='center', verticalalignment='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
        
    x_labels = [f"Turn {i+1}" for i in range(len(emotion_history))]
    frust_vals = [turn.get("frustration", 0) for turn in emotion_history]
    anger_vals = [turn.get("anger", 0) for turn in emotion_history]
    
    negative_trend = [f + a for f, a in zip(frust_vals, anger_vals)]
    
    ax.plot(x_labels, negative_trend, marker='o', color='#F44336', linewidth=2, label="Trend (Frust + Anger)")
    ax.axhline(y=0.6, color='black', linestyle='--', alpha=0.5, label="Threshold (0.6)")
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Intensidade Emocional")
    ax.set_title("Evolução Emocional da Chamada")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    return fig

# --- PROCESSAMENTO PRINCIPAL ---
def process_call_turn(audio_payload, chat_history, emotion_history):
    if audio_payload is None:
        return chat_history, emotion_history, "⚠️ Por favor, grava áudio primeiro.", plot_emotion_trend(emotion_history), None
        
    pipe = get_pipeline()
    
    # 1. Pipeline Real (Isto agora terá 100% dos recursos do PC)
    try:
        from src.demo.app import _normalise_audio 
        wav, sr = _normalise_audio(audio_payload)
        out = pipe.predict_from_audio(wav, sr=sr)
    except Exception as e:
        return chat_history, emotion_history, f"⚠️ Erro de Áudio: {str(e)}", plot_emotion_trend(emotion_history), None
    
    transcription = out.get("text", "[Empty transcription]")
    probs = out.get("meta_probs", {}) 
    
    turn_emotions = {
        "frustration": probs.get("frustration", 0),
        "anger": probs.get("anger", 0),
        "neutral": probs.get("neutral", 0)
    }
    
    chat_history.append({"role": "user", "content": transcription})
    
    emotion_history.append(turn_emotions)
    if len(emotion_history) > 3:
        emotion_history.pop(0) 
        
    # 2. Avaliar Handover
    is_handover, status_msg = check_handover_logic(emotion_history)
    
    # 3. Resposta do Bot Instantânea
    if is_handover:
        bot_reply = "I apologize for the frustration. I am transferring your call to a human agent immediately."
    else:
        bot_reply = get_instant_smart_reply(transcription)
            
    chat_history.append({"role": "assistant", "content": bot_reply})
        
    trend_plot = plot_emotion_trend(emotion_history)
    bot_audio_path = generate_bot_audio(bot_reply)
    
    return chat_history, emotion_history, status_msg, trend_plot, bot_audio_path

def reset_call():
    estado_inicial_html = "<div style='background-color:#e8f5e9; color:#2e7d32; padding:15px; border-radius:8px; text-align:center; font-weight:bold; font-size:18px; border: 2px solid #66bb6a;'>🟢 OPERADOR VIRTUAL (Ativo)</div>"
    return [], [], [], estado_inicial_html, plot_emotion_trend([]), None, None

# --- UI GRADIO ---
# Adicionamos uma cor primária personalizada e tiramos os cantos demasiado redondos
custom_theme = gr.themes.Soft(
    primary_hue="indigo", 
    secondary_hue="blue",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #333;'>🎧 SmartHandover: Demonstração Profissional</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 16px; color: #666;'>Pipeline multimodal de deteção de emoções em tempo real com handover por <i>Sliding Window</i>.</p>")
    
    chat_state = gr.State([])
    emotion_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ 1. Cliente (Voz)")
            with gr.Group(): # Agrupa os controlos de áudio numa caixa mais limpa
                audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Gravar Microfone")
                btn_submit = gr.Button("Enviar Interação", variant="primary", size="lg")
            
            gr.Markdown("### 🤖 2. Resposta do Sistema")
            bot_audio_out = gr.Audio(label="Voz do Assistente", autoplay=True, interactive=False)
            
            # Movemos o botão de Nova Chamada para baixo do áudio do bot, para fluxo mais lógico
            btn_reset = gr.Button("📞 Iniciar Nova Chamada", size="lg")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Monitorização e Decisão")
            # Substituímos a Textbox por HTML nativo para permitir as cores dinâmicas!
            estado_inicial = "<div style='background-color:#e8f5e9; color:#2e7d32; padding:15px; border-radius:8px; text-align:center; font-weight:bold; font-size:18px; border: 2px solid #66bb6a;'>🟢 OPERADOR VIRTUAL (Ativo)</div>"
            status_box = gr.HTML(value=estado_inicial, label="Estado do Handover")
            
            # Adicionamos avatares para o Utilizador e para o Bot
            chat_box = gr.Chatbot(
                label="Transcrição da Chamada", 
                avatar_images=("https://cdn-icons-png.flaticon.com/512/1077/1077114.png", "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"),
                height=550
            )
            
    with gr.Row():
        plot_out = gr.Plot(label="Sliding Window Trend")
        
    btn_submit.click(
        fn=process_call_turn,
        inputs=[audio_in, chat_state, emotion_state],
        outputs=[chat_box, emotion_state, status_box, plot_out, bot_audio_out]
    )
    
    btn_reset.click(
        fn=reset_call,
        inputs=[],
        outputs=[chat_state, chat_box, emotion_state, status_box, plot_out, bot_audio_out, audio_in]
    )

if __name__ == "__main__":
    print("[Simulador Call Center] A iniciar versão UI Avançada...")
    demo.launch(share=True)