# Model Weights & Configurations

Questa cartella contiene i checkpoint dei modelli addestrati per il progetto. Di seguito sono riportate le configurazioni specifiche utilizzate per ottenere questi pesi.

## 1. Conditional VAE
Configurazione per l'Autoencoder Variazionale Condizionato.

### Parametri di Addestramento
| Parametro | Valore | Variabile d'Ambiente Corrispondente |
| :--- | :--- | :--- |
| **Model Type** | `vae` | `MODEL_TYPE` |
| **Batch Size** | 1024 | `BATCH_SIZE` |
| **Learning Rate** | 0.0001 | `LEARNING_RATE` |

### Iperparametri del Modello
| Parametro | Valore | Variabile d'Ambiente Corrispondente |
| :--- | :--- | :--- |
| **Attribute Embedding Dim** | 128 | `ATTR_EMBED_DIM` |
| **Latent Dimension** | 512 | `LATENT_DIM` |
| **Beta (KL Weight)** | 0.5 | `BETA` |
| **Hidden Dims** | 64, 128, 256, 512 | `HIDDEN_DIMS` |

## 2. Conditional Diffusion (DDPM)
Configurazione per il Modello di Diffusione Condizionato.

### Parametri di Addestramento
| Parametro | Valore | Variabile d'Ambiente Corrispondente |
| :--- | :--- | :--- |
| **Model Type** | `diff` | `MODEL_TYPE` |
| **Batch Size** | 1024 | `BATCH_SIZE` |

### Iperparametri del Modello
| Parametro | Valore | Variabile d'Ambiente Corrispondente |
| :--- | :--- | :--- |
| **Time Encoding Size** | 256 | `TIME_ENCODING_SIZE` |
| **Noise Schedule Length** | 1000 | `NOISE_SCHEDULE_L` |
| **Lambda (Guidance Scale)** | 3.0 | `LAMBDA` |

# TODO
## 3. Vision Mamba 