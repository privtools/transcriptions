# [PoC] Herramienta resumen de ponencias y eventos [PoC]

Basado en faster-whisper, LLM Mistral y Gradio

## Instrucciones

1. Edita el archivo .env con los ajustes correspondientes. <code>WHISPER_DEVICE=cuda</code> para aprovechar la GPU.
1. Para crear entorno virtual: <code>python -m venv venv</code> y Activarlo: <code>source venv/bin/activate</code>
1. Instalar dependencias: <code>pip install -r requirements.txt</code>
1. Ejecutar: <code>python main.py</code>
1. Para levantar todo chatbot (desde la carpeta docker):
   ```bash
   sudo docker compose build --no-cache
   sudo docker compose up -d
1. Para levantar nginx (desde la carpeta docker/nginx)
   ```bash
   sudo docker compose build --no-cache
   sudo docker compose up -d
