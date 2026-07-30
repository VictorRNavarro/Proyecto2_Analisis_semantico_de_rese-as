import os
import csv
import time
import random

import requests
from langdetect import detect, LangDetectException

API_KEY = os.environ.get("SERPAPI_KEY", "d4bbcc06e2999fb29920fc897332ce909916045fef159cc27bffbbcdb71d55f6")
SERPAPI_URL = "https://serpapi.com/search"

OUTPUT_CSV = "resenas_costa_rica_scraped.csv"
COLUMNAS = ["texto", "calificacion", "tipo_lugar", "fuente", "fecha", "lugar"]

TARGET_TOTAL = 3000
CHECKPOINT_EVERY = 20
DELAY_MIN, DELAY_MAX = 1.0, 2.0
MAX_PAGINAS_POR_LUGAR = 5

LIMITE_MENSUAL = 240  # cuota gratis de SerpApi, con margen de seguridad
CREDITOS_USADOS = 0

PROVINCIAS = ["San Jose", "Guanacaste", "Puntarenas", "Alajuela",
              "Cartago", "Heredia", "Limon"]

CONSULTAS = []
for prov in PROVINCIAS:
    CONSULTAS.append((f"parques nacionales en {prov} Costa Rica", "parque"))
    CONSULTAS.append((f"hoteles en {prov} Costa Rica", "hotel"))
    CONSULTAS.append((f"restaurantes en {prov} Costa Rica", "restaurante"))
random.shuffle(CONSULTAS)

session = requests.Session()


def hay_creditos():
    return CREDITOS_USADOS < LIMITE_MENSUAL


def gastar_credito():
    global CREDITOS_USADOS
    CREDITOS_USADOS += 1


def pedir(params):
    """Hace el request a SerpApi, con reintento si llega un 429."""
    while True:
        r = session.get(SERPAPI_URL, params=params, timeout=15)
        gastar_credito()
        if r.status_code == 429:
            print("  [429] esperando 15s...")
            time.sleep(15)
            continue
        r.raise_for_status()
        return r.json()


def buscar_lugares(query):
    if not hay_creditos():
        return []
    params = {"engine": "google_maps", "q": query, "type": "search",
              "hl": "es", "api_key": API_KEY}
    try:
        return pedir(params).get("local_results", [])
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] busqueda '{query}': {e}")
        return []


def obtener_resenas(data_id, max_paginas=MAX_PAGINAS_POR_LUGAR):
    resenas = []
    next_page_token = None

    for _ in range(max_paginas):
        if not hay_creditos():
            break
        params = {"engine": "google_maps_reviews", "data_id": data_id,
                  "hl": "es", "api_key": API_KEY}
        if next_page_token:
            params["next_page_token"] = next_page_token

        try:
            data = pedir(params)
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] reseñas de {data_id}: {e}")
            break

        resenas.extend(data.get("reviews", []))
        next_page_token = data.get("serpapi_pagination", {}).get("next_page_token")
        if not next_page_token:
            break
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return resenas


def es_espanol(texto):
    if not texto or len(texto.strip()) < 10:
        return False
    try:
        return detect(texto) == "es"
    except LangDetectException:
        return False


def guardar_checkpoint(filas, path=OUTPUT_CSV):
    existe = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        if not existe:
            writer.writeheader()
        writer.writerows(filas)


def main():
    if API_KEY == "d4bbcc06e2999fb29920fc897332ce909916045fef159cc27bffbbcdb71d55f6":
        print("Falta configurar SERPAPI_KEY")
        return

    vistos_data_ids = set()
    vistos_textos = set()
    buffer = []
    total_guardadas = 0
    lugares_procesados = 0

    print(f"Meta: {TARGET_TOTAL} reseñas en español")

    for query, tipo_lugar in CONSULTAS:
        if total_guardadas >= TARGET_TOTAL or not hay_creditos():
            break

        print(f"Buscando: {query} [{tipo_lugar}] (creditos: {CREDITOS_USADOS}/{LIMITE_MENSUAL})")
        lugares = buscar_lugares(query)
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        for lugar in lugares:
            if total_guardadas >= TARGET_TOTAL or not hay_creditos():
                break

            data_id = lugar.get("data_id")
            nombre_lugar = lugar.get("title", "")
            if not data_id or data_id in vistos_data_ids:
                continue
            vistos_data_ids.add(data_id)

            resenas = obtener_resenas(data_id)
            lugares_procesados += 1

            for res in resenas:
                texto = (res.get("snippet") or "").strip()
                if not texto or texto in vistos_textos or not es_espanol(texto):
                    continue

                vistos_textos.add(texto)
                buffer.append({
                    "texto": texto,
                    "calificacion": res.get("rating"),
                    "tipo_lugar": tipo_lugar,
                    "fuente": "google_maps",
                    "fecha": res.get("date", ""),
                    "lugar": nombre_lugar,
                })
                total_guardadas += 1

            if lugares_procesados % CHECKPOINT_EVERY == 0 and buffer:
                guardar_checkpoint(buffer)
                print(f"  [checkpoint] {total_guardadas} reseñas ({lugares_procesados} lugares)")
                buffer = []

    if buffer:
        guardar_checkpoint(buffer)

    print(f"\nListo. Reseñas nuevas: {total_guardadas}")
    print(f"Lugares procesados: {lugares_procesados}")
    print(f"Creditos usados: {CREDITOS_USADOS}/{LIMITE_MENSUAL}")
    print(f"Archivo: {OUTPUT_CSV}")

    if total_guardadas < TARGET_TOTAL:
        print("\nNo se llego a la meta con la cuota de este mes.")
        print("El checkpoint no se borra, se puede seguir acumulando despues.")


if __name__ == "__main__":
    main()
