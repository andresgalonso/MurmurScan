

# MurmurScan

### Heart Sound Classification using Machine Learning

---

Universidad Autónoma de Chihuahua  
Facultad de Ingeniería · Data Science  
Proyecto Final · Mayo 2026  

Olanda Prieto Ordaz  
Andres Gonzalez Alonso  



---

# Descripción General

MurmurScan es una aplicación desarrollada en Streamlit para la clasificación de sonidos cardíacos mediante modelos de Machine Learning.

El sistema analiza archivos de audio `.wav` correspondientes a fonocardiogramas y predice si el sonido cardíaco es:

- Normal
- Anormal

El proyecto integra procesamiento de señales, extracción de características acústicas y modelos de clasificación supervisada.

---

# Base de Datos

Se utilizó la base de datos:

## PhysioNet Challenge 2022
### CirCor DigiScope Phonocardiogram Dataset

La base contiene grabaciones reales de sonidos cardíacos junto con información clínica de los pacientes.

Variables utilizadas:

- Edad
- Peso
- Altura
- Estado de embarazo
- Ubicaciones de grabación
- Pistas de audio

---

# Modelos Utilizados

Los modelos implementados fueron:

- Support Vector Machine 
- Random Forest
- Stacking Ensemble

---

# Procesamiento de Audio

Los audios fueron procesados utilizando técnicas de extracción de características espectrales.

Configuración:

- Frecuencia de muestreo: 4000 Hz
- Duración máxima: 10 segundos

Características extraídas:

- MFCC
- Delta MFCC
- Chroma Features
- Spectral Centroid
- Spectral Rolloff
- Zero Crossing Rate
- RMS Energy

---

# Variables Clínicas Consideradas

El sistema incorpora variables tabulares relacionadas con el paciente:

- Edad categórica
- Altura
- Peso
- IMC
- Estado de embarazo
- Número de ubicaciones registradas

---

# Visualización

- Forma de onda
- MFCC
- Energía RMS
- Zero Crossing Rate

---

# Tecnologías Utilizadas

- Python
- Streamlit
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Matplotlib
- Joblib

---

# Estructura del Proyecto

```bash
MurmurScan/
│
├── app.py
├── requirements.txt
├── models/
├── training_data/
└── README.md
```

---

# Instalación

## Clonar repositorio

```bash
git clone https://github.com/andresgalonso/MurmurScan.git
cd MurmurScan
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar aplicación

```bash
streamlit run app.py
```

---

