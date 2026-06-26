# Medição de Plântulas de Alface

Trabalho de Processamento de Imagens / Computação Gráfica (UNISC).
Mede digitalmente o comprimento das estruturas de plântulas de alface a partir
de uma foto, usando **OpenCV**.

Para cada plântula, mede dois segmentos **seguindo o caminho real (curvo)** da
estrutura — não em linha reta:

- **Segmento 1:** topo da estrutura branca → ponto de estrangulamento
- **Segmento 2:** ponto de estrangulamento → extremidade final da raiz
- **Total:** soma dos dois segmentos

Os resultados saem em **centímetros** (calibrados pela régua da foto) numa
**tabela (CSV)** e numa **imagem anotada** com os três pontos e os dois
segmentos marcados.

## Como funciona (pipeline)

O grande desafio é que o filamento e a raiz são **brancos sobre papel branco**
(quase sem contraste num limiar direto). A solução:

1. **ROI do papel** (`src/structure.py`): isola o quadrado de papel branco,
   descartando régua, bordas da caixa e mesa.
2. **Realce por subtração de fundo**: `cinza - desfoque_grande` faz os
   filamentos finos aparecerem como linhas claras sobre fundo escuro
   (estruturas brancas viram alto contraste).
3. **Mapa de custo + caminho de menor custo** (`src/trace.py`): sobre o realce,
   o filamento tem custo baixo. O caminho ótimo
   (`skimage.graph.route_through_array`) entre o topo e a ponta **acompanha a
   curva real da raiz**, mesmo enrolada.
4. **Pontos-chave**: topo (cotilédone/semente), estrangulamento (transição
   hipocótilo→raiz, estimada pela variação de espessura via *distance
   transform*) e ponta (ponto geodesicamente mais distante).
5. **Medição curvilínea**: soma das distâncias ao longo do caminho suavizado,
   convertida para cm pela escala.
6. **Calibração** (`src/calibration.py`): lê a régua automaticamente
   (espaçamento das marcas de cm) **ou** por dois cliques sobre marcas
   conhecidas.

Como as raízes brancas se cruzam/encostam, a detecção 100% automática pode
errar em alguns casos. Por isso o programa é **híbrido**: roda o automático e
permite **corrigir com cliques**.

## Requisitos

- Python 3.10+
- Dependências em `requirements.txt` (OpenCV, NumPy, scikit-image, SciPy,
  Pillow + pillow-heif para abrir `.HEIC`).

## Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Uso

### Modo híbrido (recomendado): automático + correção manual

```bash
python main.py --imagem data/entrada/IMG_3265.png
```

Abre uma janela com as plântulas já detectadas. Atalhos:

| Tecla | Ação |
|-------|------|
| `a` | adicionar plântula: clique no **topo** (cotilédone) e depois na **ponta** da raiz — o caminho curvo é traçado automaticamente |
| `d` | apagar: clique sobre uma plântula |
| `c` | ajustar o **estrangulamento**: clique no ponto correto sobre a plântula |
| `k` | calibrar por cliques: clique em 2 marcas de cm e informe a distância no console |
| `r` | re-rodar a detecção automática |
| `z` | desfazer última adição |
| `s` | salvar (imagem anotada + CSV + estado) |
| `h` | mostrar/ocultar ajuda |
| `q` / `ESC` | sair (salvando) |

### Modo automático (sem janela)

```bash
python main.py --imagem data/entrada/IMG_3265.png --sem-gui
```

### Opções úteis

```bash
# forçar a escala (px/cm na imagem original), ignorando a régua
python main.py --imagem foto.png --px-por-cm 195

# calibrar por cliques desde o início (na janela)
python main.py --imagem foto.png --calibracao cliques

# trocar a pasta de saída
python main.py --imagem foto.png --saida data/saida
```

Aceita imagens `.HEIC`, `.PNG`, `.JPG` diretamente.

## Saídas

Geradas em `data/saida/`:

- `<nome>_anotado.png` — imagem com os 3 pontos (topo, estrangulamento, ponta),
  os 2 segmentos em cores distintas, IDs e a tabela de medidas;
- `<nome>.csv` — tabela: plântula, seg1, seg2, total (cm e px), origem;
- `<nome>_estado.json` — estado para reabrir/editar depois.

Legenda das marcações:

- ● amarelo: topo da estrutura branca
- ● vermelho: ponto de estrangulamento
- ● magenta: extremidade da raiz
- linha verde: segmento 1 (topo → estrangulamento)
- linha azul: segmento 2 (estrangulamento → ponta)

## Estrutura do projeto

```
medir-plantula/
├── main.py                 # CLI (modo hibrido e headless)
├── requirements.txt
├── src/
│   ├── io_utils.py         # leitura/escrita (suporte a HEIC)
│   ├── structure.py        # ROI do papel + realce + mascara/custo
│   ├── trace.py            # caminho de menor custo + medicao + estrangulamento
│   ├── detect.py           # deteccao automatica (cabecas + pontas)
│   ├── calibration.py      # escala: regua automatica + cliques
│   ├── visualize.py        # marcacoes e tabela na imagem
│   ├── export.py           # CSV + estado JSON
│   ├── pipeline.py         # orquestracao na resolucao de trabalho
│   └── interactive.py      # editor com correcao por cliques (GUI)
├── tools/                  # scripts de exploracao/conversao (apoio)
└── data/
    ├── entrada/            # imagens de entrada
    └── saida/              # resultados
```

## Atendimento aos critérios de avaliação

- **Identificação das estruturas:** topo, estrangulamento e ponta marcados.
- **Divisão em 2 segmentos:** seg1 (topo→estrangulamento) e seg2
  (estrangulamento→raiz).
- **Precisão / caminho real:** medição ao longo do caminho de menor custo
  (acompanha curvas e partes enroladas), não em linha reta.
- **Escala:** régua automática ou calibração por cliques → cm.
- **Organização:** tabela CSV + tabela na imagem.
- **Clareza visual:** pontos e segmentos marcados na imagem.
- **OpenCV:** usado em todo o processamento.

## Limitações conhecidas

- A leitura automática da régua assume marcas de cm regulares e mais altas que
  as de mm; se falhar, use a calibração por 2 cliques (`k`).
- A detecção automática pode não pegar todas as plântulas (raízes que se cruzam,
  cotilédones pouco visíveis). Use o modo híbrido (`a`) para completar.
- A régua fica na borda elevada da caixa (plano um pouco mais próximo da câmera
  que o papel), o que introduz um pequeno erro de perspectiva na escala.
