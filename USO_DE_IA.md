# Uso de IA - Proyecto 2: Análisis Semántico de Reseñas Turísticas

## Herramientas utilizadas

- **Codex (OpenAI):** apoyo para revisar la rúbrica, organizar y depurar los análisis de Word2Vec y BETO, mejorar el dashboard en Plotly Dash y verificar la interpretación de las métricas.
- **Claude (Anthropic):** apoyo para investigar y comparar alternativas de web scraping, librerías y estrategias de extracción de reseñas.

## Uso de Claude en web scraping

Se utilizó Claude de Anthropic para investigar y comparar distintas opciones de web scraping. La herramienta ayudó a identificar librerías, enfoques alternativos para extraer información de sitios web y ejemplos de código. Esto permitió evaluar alternativas y seleccionar un enfoque adecuado para recopilar reseñas turísticas, considerando el uso responsable de solicitudes, límites de API y almacenamiento posterior en MongoDB.

## Ejemplos de prompts utilizados

1. "Revisa este proyecto de análisis semántico y compáralo con los requisitos de Word2Vec, BETO y Plotly Dash."
2. "Ayúdame a corregir el dashboard para que cargue el corpus disponible y muestre resultados aunque algunos artefactos aún no existan."
3. "¿Cómo puedo comparar BoW, Word2Vec y BETO con clasificación, clustering y visualización t-SNE?"
4. "Compara opciones de web scraping para obtener reseñas turísticas, incluyendo librerías, límites de solicitudes y almacenamiento en MongoDB."
5. "Explica cómo manejar errores, pausas entre solicitudes y credenciales de API en un proceso de recopilación de reseñas."

## Reflexión sobre el uso de IA

La IA se utilizó como una herramienta de apoyo para comprender técnicas de procesamiento de lenguaje natural, evaluar opciones de implementación y detectar problemas técnicos. No se utilizó como sustituto del análisis: los modelos se ejecutaron sobre el corpus del proyecto y los resultados se revisaron antes de interpretarlos.

También se verificó que una métrica alta no siempre implica un resultado excelente. Por ejemplo, el corpus tiene más reseñas positivas y de parques que otras categorías; por ello, las métricas de clasificación y clustering se interpretaron considerando el desbalance de clases y no solo la accuracy.

## Cambios y decisiones realizados por el equipo

- Se adaptó el entrenamiento de Word2Vec al tamaño del corpus y se añadieron centroides por tipo de lugar y polaridad.
- Se implementó el análisis de polisemia de BETO usando el embedding contextual del token, en lugar de comparar únicamente el vector de la oración completa.
- Se incorporó clasificación por tipo de lugar, evaluación de alineación de clusters y una visualización t-SNE comparativa.
- Se mejoró el dashboard con resultados de Word2Vec, BETO, comparación de representaciones y un tema visual relacionado con turismo en Costa Rica.
- Se revisaron alternativas para recopilar reseñas y se documentó la necesidad de respetar límites de solicitudes, credenciales y buenas prácticas de scraping.

También se verificó que una métrica alta no siempre implica un resultado excelente. Por ejemplo, el corpus tiene más reseñas positivas y de parques que otras categorías; por ello, las métricas de clasificación y clustering se interpretaron considerando el desbalance de clases y no solo la accuracy.
