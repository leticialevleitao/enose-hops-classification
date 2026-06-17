# Classificação Inteligente de Variedades de Lúpulo Utilizando Nariz Eletrônico e Modelos de Aprendizado de Máquina

Repositório oficial do código, dos dados e dos modelos do Trabalho de Conclusão de Curso (TCC) de **Letícia de Albuquerque S. Leitão**, do curso de Sistemas de Informação do Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE), sob orientação do **Prof. Dr. Leandro Maciel Almeida**.

O trabalho dá continuidade à linha de pesquisa do grupo **CIn.AI**, conduzida em parceria com o **CETENE** (Centro de Tecnologias Estratégicas do Nordeste), voltada ao desenvolvimento de um nariz eletrônico (e-nose) portátil para classificação automática de variedades de lúpulo a partir da análise de compostos orgânicos voláteis (COVs).

## Visão geral

O sistema utiliza um nariz eletrônico equipado com sete sensores MOS (óxido metálico semicondutor) da série TGS (Figaro Engineering Inc.), complementados por sensores ambientais de temperatura, pressão e umidade. A pipeline de aprendizado de máquina foi reformulada em relação à versão inicial do projeto para incluir:

1. **Validação rigorosa por sessão** (Leave-One-Session-Out, LOSO), corrigindo o vazamento de dados (data leakage) presente em divisões aleatórias.
2. **Extração de 378 características temporais por amostra** (estatísticas, derivadas, áreas sob a curva, ressonância e características de transitório por sensor).
3. **Comparação ampla de modelos**, contemplando modelos clássicos, gradient boosting modernos (XGBoost, LightGBM, CatBoost), modelos tabulares de fundação (TabPFN) e modelos para séries temporais.
4. **Análise não supervisionada** (clustering) para investigar a estrutura natural dos dados.
5. **Explicabilidade** com SHAP e importância por permutação, identificando os sensores mais discriminativos. O poder discriminatório individual de cada sensor é quantificado pela métrica η² (eta-quadrado, razão entre variância entre classes e variância total), com destaque para TGS822 (η² = 0,985) e TGS826 (η² = 0,971).
6. **Quantificação de Concept Drift / Covariate Shift** entre as três formas de apresentação do lúpulo (cones, pellets e óleos), evidenciando empiricamente que modelos treinados em uma forma não generalizam para as outras.
7. **Arquitetura de inferência modular com roteamento dinâmico**: um classificador gateway identifica a forma da amostra recebida e direciona a inferência ao modelo especialista correspondente, viabilizando a incorporação futura de novas variedades e novas formas sem reescrever todo o sistema.

## Arquitetura do pipeline em produção

```
                          ┌───────────────────────┐
   amostra (e-nose) ───▶ │ Gateway: detector da   │
                          │ forma (LDA, acc 91,7%) │
                          └────────────┬───────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       ┌────────────┐           ┌────────────┐           ┌────────────┐
       │Especialista│           |Especialista│           │Especialista│
       │  Cones     │           │   Pellets  │           │   Óleos    │
       │(LDA/TabPFN)│           │   (LDA)    │           │   (LDA)    │
       └────────────┘           └────────────┘           └────────────┘
        variedade                variedade                variedade
```

A motivação para essa arquitetura é discutida na seção de drift do TCC: a transferência direta de um modelo treinado em cones para amostras de pellets cai para cerca de 20% de acurácia, evidenciando o impacto do processamento industrial sobre a assinatura olfativa.

## Estrutura do repositório

```
enose-hops-classification/
├── data/
│   ├── consolidated/                Conjuntos de dados consolidados (CSV) + metadados
│   └── README.md                    Documentação do dataset (protocolo, formato)
├── enose/                           Pacote Python com a lógica do pipeline
│   ├── data_loader.py               Definições de colunas dos sensores e carregamento
│   ├── preprocessor.py              Normalização, splits e extração de features
│   ├── session_loader.py            Carregamento por sessão (necessário para LOSO)
│   ├── loso_cv.py                   Implementação de Leave-One-Session-Out CV
│   └── models/
│       ├── classical_models.py      Classificadores clássicos com otimização bayesiana
│       └── time_series_models.py    Classificadores para séries temporais (ROCKET, etc.)
├── models/                          Modelos finais serializados (.joblib)
│   ├── form_detector_lda.joblib     Gateway: detector da forma
│   ├── cone_variety_lda.joblib      Especialista: variedade de cones
│   ├── pellet_variety_lda.joblib    Especialista: variedade de pellets
│   ├── oil_variety_lda.joblib       Especialista: variedade de óleos
│   └── metadata.json                Metadados dos modelos (acurácia LOSO, n_features, etc.)
├── run_loso_experiments.py          Pipeline LOSO com modelos clássicos
├── run_loso_modern.py               Pipeline LOSO com modelos modernos (TabPFN, XGB, etc.)
├── run_production_pipeline.py       Avaliação ponta a ponta do roteamento dinâmico
├── run_clustering.py                Análise não supervisionada
├── run_eda.py                       Análise exploratória (PCA, t-SNE, curvas)
├── run_explainability.py            SHAP + importância por permutação
├── run_expanded.py                  Experimentos expandidos com formas adicionais
├── run_hop_experiments.py           Pipeline com split aleatório (baseline)
├── save_models.py                   Treinamento e serialização dos modelos finais
├── compute_metrics_table.py         Consolidação de métricas (acurácia, F1, G-mean)
├── generate_sensor_curves.py        Curvas médias de resposta por classe
├── generate_statistical_analysis.py Bootstrap CI + testes de McNemar
├── requirements.txt                 Dependências Python
├── LICENSE                          Licença MIT
└── README.md                        Este arquivo
```

## Dataset

O conjunto de dados utilizado neste TCC contém **30 arquivos CSV consolidados** (~2,3 MB) abrangendo três formas de apresentação do lúpulo: **cones**, **pellets** (T90) e **óleos** (extratos com CO₂ supercrítico). As variedades incluem Cascade, Chinook, Comet, Nugget, Saaz, Triple Pearl, Triumph, Vista e Zeus.

Cada arquivo CSV contém ciclos de medição com leituras dos sete sensores MOS (TGS) e dos sensores ambientais (temperatura, pressão, umidade). O arquivo `data/consolidated/consolidation_metadata.json` descreve a correspondência entre ciclos e sessões de coleta, informação essencial para a validação por sessão (LOSO).

Para detalhes do protocolo de coleta, formato dos CSVs e organização dos rótulos, consulte [`data/README.md`](data/README.md).

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/leticialevleitao/enose-hops-classification.git
cd enose-hops-classification
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Reproduzir os experimentos principais

```bash
# Pipeline LOSO com modelos clássicos (LDA, SVM, RF, KNN, etc.)
python run_loso_experiments.py

# Pipeline LOSO com modelos modernos (TabPFN, XGBoost, LightGBM, CatBoost)
python run_loso_modern.py

# Avaliação ponta a ponta da arquitetura de roteamento dinâmico
python run_production_pipeline.py

# Análises complementares
python run_eda.py
python run_clustering.py
python run_explainability.py
```

### 4. Inferência com os modelos treinados

Os modelos finais estão serializados em `models/` e podem ser carregados diretamente:

```python
import joblib

gateway = joblib.load("models/form_detector_lda.joblib")
specialists = {
    "cone":   joblib.load("models/cone_variety_lda.joblib"),
    "pellet": joblib.load("models/pellet_variety_lda.joblib"),
    "oil":    joblib.load("models/oil_variety_lda.joblib"),
}

# X: vetor de 378 features extraídas conforme enose/preprocessor.py
forma = gateway.predict(X)[0]
variedade = specialists[forma].predict(X)[0]
print(f"Forma identificada: {forma} | Variedade: {variedade}")
```

> O modelo TabPFN treinado para cones (acurácia LOSO de 82,4%) não está versionado no repositório por restrições de tamanho do GitHub (acima de 100 MB). Ele pode ser regenerado executando `python save_models.py`.

## Resumo dos resultados (LOSO)

| Cenário                                  | Modelo  | Acurácia | F1-score | G-mean |
|------------------------------------------|---------|----------|----------|--------|
| Detecção da forma (gateway, 3 classes)   | LDA     | 91,7%    | 0,914    | 0,931  |
| Variedade de cones (9 classes)           | TabPFN  | 82,4%    | 0,818    | 0,852  |
| Variedade de cones (9 classes)           | LDA     | 70,0%    | 0,695    | 0,742  |
| Variedade de pellets (9 classes)         | LDA     | 93,3%    | 0,930    | 0,948  |
| Variedade de óleos (12 classes)          | LDA     | 95,0%    | 0,948    | 0,961  |
| Transferência direta cones → pellets     | LDA     | ~20%     | ~0,18    | ~0,22  |

Os valores consolidados estão disponíveis em `models/metadata.json`.

## Trabalhos futuros

- Integração do pipeline com a plataforma do grupo (deploy efetivo).
- Coleta de novas sessões para ampliar a base de pellets e óleos.
- Inclusão de novas variedades e novas formas de apresentação do lúpulo.


## Licença

Este repositório está licenciado sob a [MIT License](LICENSE).
