# Lista 1: Geração de Números Pseudo-Aleatórios

**Métodos Computacionais Intensivos para Mineração de Dados**

**Departamento de Ciência da Computação — Universidade de Brasília**

| | |
|---|---|
| **Professor** | Guilherme Rodrigues |
| **Aluno** | Matheus Firetti |
| **Data** | 18 de Junho de 2026 |

---

## Uso de LLMs

<!-- Seção obrigatória — Instrução 8 -->

O uso de LLMs neste trabalho foi feito exclusivamente como ferramenta de apoio, conforme descrito abaixo:

- **Gemini** foi utilizado na **Questão 1** para auxiliar na estruturação e documentação do código original (q1.py), por meio de comentários e docstrings, otimização das rotinas de simulação (vetorização com NumPy) visando eficiência computacional, e melhorias visuais dos gráficos e dos resultados impressos no terminal.

- **Claude** foi utilizado na **Questão 2** para auxiliar na estruturação e documentação do código original (q2.py), por meio de comentários e docstrings, e melhorias visuais dos gráficos e dos resultados impressos no terminal. Também foi utilizado para auxiliar na formatação e estruturação deste relatório, e auxílio na compreensão matemática acerca da distribuição Cauchy, Transformada Inversa e Método de Aceitação-Rejeição.

---

## Questão 1 — Simulando Computacionalmente o Gerador de Babel

A Biblioteca de Babel, proposta por Jorge Luís Borges, é composta por um número infinito de galerias contendo todos os livros possíveis. A maior parte dos livros não tem qualquer significado, porém certos textos resultam em grandes obras. Nesta questão, simulamos computacionalmente geradores de sequências de 5 letras e investigamos a probabilidade de gerar palavras válidas da língua portuguesa.

**Premissas adotadas:**

- Utiliza-se somente as 26 letras do alfabeto, sem caracteres especiais.
- Não há distinção entre letras maiúsculas e minúsculas.
- O dicionário de palavras válidas é representado pelo arquivo `Dicionario.txt`.

### Preparação dos Dados

O primeiro passo é carregar o dicionário, remover acentos, normalizar para letras minúsculas, e extrair o conjunto de palavras válidas com exatamente 5 letras.

```python
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from collections import Counter
import string
import time

def remove_acentos(texto):
    """Remove acentos de palavras e normaliza."""
    texto = unicodedata.normalize('NFD', texto)
    return ''.join(ch for ch in texto if unicodedata.category(ch) != 'Mn')

caminho_dicionario = "./Dicionario.txt"
palavras_validas = set()
letras_dicionario = []

with open(caminho_dicionario, "r", encoding="utf-8", errors="ignore") as f:
    for linha in f:
        palavra = linha.strip().lower()
        palavra_limpa = remove_acentos(palavra)
        if palavra_limpa.isalpha():
            letras_dicionario.extend(list(palavra_limpa))
            if len(palavra_limpa) == 5:
                palavras_validas.add(palavra_limpa)

qtd_validas = len(palavras_validas)
alfabeto = list(string.ascii_lowercase)
espaco_amostral_a = 26**5

print(f"Total de palavras válidas (5 letras) no dicionário: {qtd_validas}")
```

**Output:** Total de palavras válidas (5 letras) no dicionário: **5.427**

### Item (a) — Probabilidade de Gerar Palavra Válida

#### Cálculo Analítico

O espaço amostral total é 26⁵ = 11.881.376 sequências possíveis. Sendo *k* o número de palavras válidas de 5 letras no dicionário, a probabilidade analítica é:

$$P(\text{palavra válida}) = \frac{k}{26^5}$$

#### Simulação Monte Carlo

Geramos N = 1.000.000 sequências aleatórias de 5 letras com probabilidade uniforme e verificamos quantas são palavras válidas do dicionário.

```python
np.random.seed(42)
N_max = 1000000

seq_indices = np.random.randint(0, 26, size=(N_max, 5))
seq_letras = np.array(alfabeto)[seq_indices]
palavras_geradas = [''.join(seq) for seq in seq_letras]
validades = np.array([1 if p in palavras_validas else 0 for p in palavras_geradas])

# Probabilidades
prob_analitica_a = qtd_validas / espaco_amostral_a
p_estimada_a = validades.sum() / N_max
```

#### Gráfico de Convergência

O gráfico abaixo mostra como a estimativa de Monte Carlo converge para o valor teórico conforme o tamanho da amostra aumenta. O intervalo de confiança de 95% (baseado no Teorema Central do Limite) forma um "funil" ao redor do valor teórico:

```python
tamanhos_amostra = np.arange(1000, N_max+1, 2000)
est_cumulativas = [np.mean(validades[:n]) for n in tamanhos_amostra]

Z = 1.96
erro_padrao = np.sqrt(prob_analitica_a * (1 - prob_analitica_a) / tamanhos_amostra)
limite_superior = prob_analitica_a + Z * erro_padrao
limite_inferior = prob_analitica_a - Z * erro_padrao

plt.figure(figsize=(10, 6))
plt.plot(tamanhos_amostra, est_cumulativas, label="Estimativa (Monte Carlo)", color="blue")
plt.axhline(y=prob_analitica_a, color="red", linestyle="--", label="Valor Teórico")
plt.plot(tamanhos_amostra, limite_superior, color="gray", linestyle=":",
         label="Intervalo de Confiança (95%)")
plt.plot(tamanhos_amostra, limite_inferior, color="gray", linestyle=":")
plt.fill_between(tamanhos_amostra, limite_inferior, limite_superior, color="gray", alpha=0.2)

plt.title("Item A: Convergência da Probabilidade de Gerar Palavra Válida")
plt.xlabel("Tamanho da Amostra (N)")
plt.ylabel("Probabilidade Estimada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_convergencia_item_1a.png", dpi=300, bbox_inches='tight')
plt.close()
```

![Convergência da Estimativa Monte Carlo — Item (a)](grafico_convergencia_item_1a.png)

#### Resultados e Discussão

| Medida | Valor |
|--------|-------|
| Palavras válidas (5 letras) no dicionário | 5.427 |
| Espaço amostral (26⁵) | 11.881.376 |
| Probabilidade Analítica (5427 / 26⁵) | 0,000457 (0,0457%) |
| Probabilidade Estimada (Monte Carlo, N=10⁶) | 0,000497 (0,0497%) |

O gráfico de convergência mostra que a estimativa Monte Carlo se aproxima da região do valor teórico conforme N cresce, com o intervalo de confiança de 95% estreitando-se a uma taxa O(1/√N). Entretanto, a estimativa final (0,0497%) ficou ligeiramente acima do valor analítico (0,0457%), uma diferença de aproximadamente 0,004 pontos percentuais. Para avaliar se essa discrepância é compatível com flutuação amostral, calculamos o erro padrão da estimativa:

$$\text{EP} = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0{,}000457 \times 0{,}999543}{10^6}} \approx 2{,}14 \times 10^{-5}$$

A diferença observada corresponde a (0,000497 − 0,000457) / 0,0000214 ≈ **1,87 desvios padrão**, valor que se encontra dentro do intervalo de confiança de 95% (±1,96σ). Portanto, a flutuação observada é estatisticamente compatível com o valor teórico, embora esteja próxima ao limite superior do intervalo. Visando uma melhor aproximação, seguindo a Lei dos Grandes Números, foi possível aumentar o número de sequências para, apenas, 10⁷, com um custo computacional elevado. Com amostras ainda maiores (N = 10⁸, 10⁹), a estimativa tende a se aproximar mais do valor analítico.

---

### Item (b) — Probabilidade de Palíndromos

#### Cálculo Analítico

Um palíndromo de 5 letras tem o formato L₁L₂L₃L₂L₁, com 3 graus de liberdade independentes (L₁, L₂, L₃). O número total de palíndromos possíveis é 26³, logo:

$$P(\text{palíndromo}) = \frac{26^3}{26^5} = \frac{1}{26^2} = \frac{1}{676} \approx 0{,}001479$$

#### Simulação Monte Carlo

Reutilizamos as N = 1.000.000 sequências já geradas no item (a), verificando quais são palíndromos (isto é, a string é igual à sua inversa).

```python
val_palindromo = np.array([1 if p == p[::-1] else 0 for p in palavras_geradas])
prob_estimada_b = val_palindromo.sum() / N_max
```

#### Resultados e Discussão

| Medida | Valor |
|--------|-------|
| Probabilidade Analítica (1/676) | 0,001479 (0,1479%) |
| Probabilidade Estimada (Monte Carlo, N=10⁶) | 0,001527 (0,1527%) |

Os resultados da simulação confirmam a probabilidade teórica, demonstrando que a proporção de palíndromos é muito pequena, aproximadamente 0,15% das sequências aleatórias de 5 letras.

---

### Item (c) — Gerador Alternado Consoante/Vogal

#### Descrição do Gerador

Este gerador impõe uma alternância entre consoantes e vogais. A primeira letra é sorteada uniformemente entre as 26 letras. Se for uma vogal, a próxima será uma consoante e vice-versa, formando os padrões:

- Se a primeira letra é vogal: **V–C–V–C–V** (5 × 21 × 5 × 21 × 5)
- Se a primeira letra é consoante: **C–V–C–V–C** (21 × 5 × 21 × 5 × 21)

#### Simulação Monte Carlo

```python
vogais = ['a', 'e', 'i', 'o', 'u']
consoantes = [c for c in alfabeto if c not in vogais]

N_c = 1000000
primeira_letra_char = np.random.choice(alfabeto, size=N_c)

# Apenas 4 posições restantes, pois a primeira letra já foi sorteada
escolhas_vogais = np.random.choice(vogais, size=(N_c, 4))
escolhas_consoantes = np.random.choice(consoantes, size=(N_c, 4))

is_vogal = np.isin(primeira_letra_char, vogais)

# A PRIMEIRA LETRA da sequência É a que foi sorteada uniformemente do alfabeto
# Cenário 1: V-C-V-C-V
vog = np.char.add(primeira_letra_char, escolhas_consoantes[:,0])
vog = np.char.add(vog, escolhas_vogais[:,0])
vog = np.char.add(vog, escolhas_consoantes[:,1])
vog = np.char.add(vog, escolhas_vogais[:,1])

# Cenário 2: C-V-C-V-C
cons = np.char.add(primeira_letra_char, escolhas_vogais[:,0])
cons = np.char.add(cons, escolhas_consoantes[:,0])
cons = np.char.add(cons, escolhas_vogais[:,1])
cons = np.char.add(cons, escolhas_consoantes[:,1])

geradas_c = np.where(is_vogal, vog, cons)
validas_c_arr = [1 if p in palavras_validas else 0 for p in geradas_c]
prob_estimada_c = sum(validas_c_arr) / N_c
```

#### Resultados e Discussão

| Medida | Valor |
|--------|-------|
| Probabilidade Estimada (Gerador Alternado, N=10⁶) | 0,006946 (0,6946%) |

A restrição de alternância entre consoantes e vogais altera significativamente o espaço amostral e a probabilidade de gerar palavras válidas, uma vez que muitas palavras da língua portuguesa seguem naturalmente esse padrão silábico.

---

### Item (d) — Frequências da Língua Portuguesa e Probabilidade Condicional

#### Descrição do Gerador

Neste item, cada letra é sorteada com probabilidade proporcional à sua frequência na língua portuguesa, conforme dados da [Wikipedia](https://pt.wikipedia.org/wiki/Frequ%C3%AAncia_de_letras). Estima-se a probabilidade de gerar uma palavra válida **dado que** a sequência contém ao menos um "a".

#### Frequências Utilizadas

As frequências da língua portuguesa (Wikipedia) foram normalizadas para somar exatamente 1:

```python
freqs_wikipedia_pct = {
    'a': 14.63, 'b': 1.04, 'c': 3.88, 'd': 4.99, 'e': 12.57, 'f': 1.02, 'g': 1.30,
    'h': 1.28,  'i': 6.18, 'j': 0.40, 'k': 0.02, 'l': 2.78,  'm': 4.74, 'n': 5.05,
    'o': 10.73, 'p': 2.52, 'q': 1.20, 'r': 6.53, 's': 7.81,  't': 4.34, 'u': 4.63,
    'v': 1.67,  'w': 0.01, 'x': 0.21, 'y': 0.01, 'z': 0.47
}
soma_pct = sum(freqs_wikipedia_pct.values())
freqs_wikipedia = {k: v/soma_pct for k, v in freqs_wikipedia_pct.items()}
p_wiki = [freqs_wikipedia[letra] for letra in alfabeto]
```

#### Simulação Monte Carlo com Evento Condicional

Geramos N = 1.000.000 sequências usando as frequências ponderadas e filtramos aquelas que contêm ao menos uma letra "a". A probabilidade condicional é estimada como:

$$\hat{P}(\text{válida} \mid \text{contém 'a'}) = \frac{\#\{\text{válidas com 'a'}\}}{\#\{\text{total com 'a'}\}}$$

```python
N_d = 1000000
seq_indices_d = np.random.choice(26, size=(N_d, 5), p=p_wiki)
seq_letras_d = np.array(alfabeto)[seq_indices_d]
palavras_geradas_d = [''.join(seq) for seq in seq_letras_d]

tem_a = np.array(['a' in p for p in palavras_geradas_d])
palavras_com_a = np.array(palavras_geradas_d)[tem_a]

palavras_validas_geradas_com_a = [p for p in palavras_com_a if p in palavras_validas]
prob_condicional = len(palavras_validas_geradas_com_a) / len(palavras_com_a)
```

#### Resultados e Discussão

| Medida | Valor |
|--------|-------|
| Total gerado com ao menos um "a" | 546.505 (de 1.000.000) |
| Palavras válidas dentre estas | 5.549 (2.026 únicas) |
| Probabilidade Condicional Estimada P(válida \| contém 'a') | 0,010154 (1,0154%) |

A utilização de frequências ponderadas pela distribuição real da língua portuguesa, combinada com o condicionamento à presença de ao menos um "a" (a letra mais frequente), aumenta significativamente a probabilidade de gerar palavras válidas em relação ao gerador uniforme do item (a).

**Para pensar:** O cálculo analítico dessa probabilidade condicional exigiria enumerar todas as palavras válidas de 5 letras que contêm ao menos um "a" e ponderar cada uma pela probabilidade de ser gerada sob o modelo de frequências da Wikipedia — uma tarefa computacionalmente viável mas algebricamente complexa devido à dependência posicional das letras.

---

## Questão 2 — Gerando Números Pseudo-Aleatórios

### Item (a) — Distribuição Cauchy via Transformada Inversa

#### Fundamentação Teórica

A distribuição Cauchy(γ) possui densidade:

$$f(x) = \frac{1}{\pi \gamma \left(1 + (x/\gamma)^2\right)}$$

Sua função de distribuição acumulada (CDF) é:

$$F(x) = \frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{x}{\gamma}\right)$$

Pelo método da transformada inversa, se U ~ Uniforme(0,1), então:

$$X = F^{-1}(U) = \gamma \cdot \tan\left[\pi\left(U - \frac{1}{2}\right)\right]$$

gera uma variável aleatória com distribuição Cauchy(γ).

#### Implementação

A função `gerar_cauchy` recebe o tamanho da amostra `n` e o parâmetro de escala `gamma`, gerando valores via transformada inversa de forma vetorizada com NumPy:

```python
def gerar_cauchy(n, gamma, seed=None):
    """
    Gera uma amostra aleatória de tamanho n da distribuição Cauchy(gamma)
    usando o método da transformada inversa.

    Parâmetros:
        n     (int)   : tamanho da amostra desejada
        gamma (float) : parâmetro de escala da distribuição Cauchy (gamma > 0)
        seed  (int)   : semente para reprodutibilidade (opcional)

    Retorna:
        np.ndarray : vetor de tamanho n com amostras da Cauchy(gamma)
    """
    if seed is not None:
        np.random.seed(seed)

    # Gera n valores uniformes em (0, 1)
    u = np.random.uniform(0, 1, size=n)

    # Aplica a transformada inversa da CDF da Cauchy
    x = gamma * np.tan(np.pi * (u - 0.5))

    return x
```

#### Resultados e Discussão

Demonstração com γ = 1 (Cauchy padrão) e n = 10.000:

| Estatística | Valor |
|-------------|-------|
| Mediana da amostra | −0,0235 (teórica: 0) |
| γ = 0.5: mediana | 0,0027 |
| γ = 2.0: mediana | 0,0107 |
| γ = 5.0: mediana | 0,0267 |

O histograma da amostra gerada sobreposto à densidade teórica confirma a qualidade do gerador. Destaca-se a presença de valores extremos, conforme mencionado na literatura como parte do conceito de "fat tails":

![Cauchy via Transformada Inversa](grafico_cauchy_item_2a.png)

---

### Item (b) — Distribuição Discreta via Transformada Inversa

#### Fundamentação Teórica

A variável aleatória discreta X tem a seguinte função massa de probabilidade:

| x | 2 | 3 | 5 | 6 | 9 |
|---|---|---|---|---|---|
| p(x) | 0,3 | 0,1 | 0,1 | 0,3 | 0,2 |

A CDF é: F(2) = 0,3; F(3) = 0,4; F(5) = 0,5; F(6) = 0,8; F(9) = 1,0.

Pelo método da transformada inversa discreta, gera-se U ~ Uniforme(0,1) e atribui-se X = xᵢ tal que F(xᵢ₋₁) < U ≤ F(xᵢ), onde F é a CDF.

#### Implementação

A função utiliza `np.searchsorted` para implementar a transformada inversa discreta de forma vetorizada, evitando loops:

```python
def gerar_discreta_inversa(n, mp_x, cdf, seed=None):
    """
    Gera uma amostra aleatória de tamanho n de uma distribuição discreta
    usando o método da transformada inversa.

    Parâmetros:
        n       (int)        : tamanho da amostra
        mp_x    (np.ndarray) : valores possíveis de X (ordenados)
        cdf     (np.ndarray) : CDF acumulada correspondente
        seed    (int)        : semente para reprodutibilidade (opcional)

    Retorna:
        np.ndarray : vetor de tamanho n com amostras da distribuição discreta
    """
    if seed is not None:
        np.random.seed(seed)

    u = np.random.uniform(0, 1, size=n)

    # np.searchsorted encontra, para cada u, o menor índice i tal que cdf[i] >= u
    indices = np.searchsorted(cdf, u, side="left")

    return mp_x[indices]
```

Para comparação, também geramos a mesma amostra usando `np.random.choice` (equivalente ao `sample` do R):

```python
# Método 1: Transformada Inversa
amostra_inversa = gerar_discreta_inversa(n=n_b, mp_x=mp_x, cdf=cdf_acumulada, seed=42)

# Método 2: np.random.choice (equivalente ao sample do R)
np.random.seed(123)
amostra_choice = np.random.choice(mp_x, size=n_b, p=prob)
```

#### Tabela Comparativa de Frequências

| x | p(x) Teórica | Freq. Relativa (Transf. Inversa) | Freq. Relativa (`np.random.choice`) |
|---|---|---|---|
| 2 | 0,30 | 0,3190 | 0,2870 |
| 3 | 0,10 | 0,1020 | 0,1170 |
| 5 | 0,10 | 0,0820 | 0,1040 |
| 6 | 0,30 | 0,2980 | 0,2940 |
| 9 | 0,20 | 0,1990 | 0,1980 |

#### Resultados e Discussão

Ambos os métodos produzem frequências relativas próximas das probabilidades teóricas, como esperado para n = 1.000. Os valores diferem ligeiramente entre si por utilizarem sementes aleatórias distintas (`seed=42` para a transformada inversa e `seed=123` para `np.random.choice`), confirmando que ambos convergem independentemente para a distribuição teórica. O gráfico abaixo permite a visualização comparativa:

![Comparação de Frequências — Item 2b](grafico_discreta_item_2b.png)

---

### Item (c) — Distribuição Normal via Aceitação e Rejeição

#### Fundamentação Teórica

O método de aceitação e rejeição gera amostras de uma distribuição-alvo f(x) usando uma distribuição geradora g(x) tal que f(x) ≤ M · g(x) para todo x e alguma constante M ≥ 1.

- **Distribuição-alvo**: Normal padrão — f(x) = (1/√(2π)) · e^(−x²/2)
- **Distribuição geradora**: Cauchy padrão (γ = 1) — g(x) = 1/(π(1+x²))
- **Constante envolvente**: Para determinar M, calculamos o supremo de f(x)/g(x):

$$M = \sup_x \frac{f(x)}{g(x)} = \frac{\pi}{\sqrt{2\pi}} \cdot \sup_x \left[(1+x^2) \cdot e^{-x^2/2}\right]$$

Derivando h(x) = (1+x²)·e^(−x²/2) e igualando a zero, encontramos que o máximo ocorre em x = ±1, com h(1) = 2/√e. Logo:

$$M = \sqrt{\frac{2\pi}{e}} \approx 1{,}5203$$

O algoritmo é:

1. Gerar Y ~ Cauchy(1) usando a função do item (a)
2. Gerar U ~ Uniforme(0,1)
3. Calcular a razão simplificada: r = (1 + Y²) · exp(−(Y² − 1)/2) / 2
4. Aceitar Y se U ≤ r, caso contrário rejeitar e voltar ao passo 1

A taxa de aceitação teórica é 1/M ≈ 0,6577 (≈ 65,8%).

#### Implementação

A função gera candidatos em lotes para eficiência computacional, evitando loops unitários:

```python
def gerar_normal_aceitacao_rejeicao(n, seed=None):
    """
    Gera uma amostra de tamanho n da distribuição Normal padrão (mu=0, sigma=1)
    usando o método de aceitação-rejeição com a Cauchy padrão (gamma=1).

    Parâmetros:
        n    (int) : tamanho da amostra desejada
        seed (int) : semente para reprodutibilidade (opcional)

    Retorna:
        tuple: (amostra, total_gerados, taxa_aceitacao)
    """
    if seed is not None:
        np.random.seed(seed)

    M = np.sqrt(2 * np.pi / np.e)
    amostra = []
    gerados = 0

    while len(amostra) < n:
        lote_size = int((n - len(amostra)) * M * 1.2) + 100

        # Passo 1: Gerar candidatos Y ~ Cauchy(1)
        y = gerar_cauchy(n=lote_size, gamma=1.0)

        # Passo 2: Gerar U ~ Uniforme(0, 1)
        u = np.random.uniform(0, 1, size=lote_size)

        # Passo 3: Razão simplificada f(y) / (M * g(y))
        razao = (1 + y**2) * np.exp(-(y**2 - 1) / 2) / 2

        # Passo 4: Aceitar se U <= razão
        aceitos = y[u <= razao]
        amostra.extend(aceitos.tolist())
        gerados += lote_size

    amostra = np.array(amostra[:n])
    taxa_aceitacao = n / gerados

    return amostra, gerados, taxa_aceitacao
```

#### Resultados e Discussão

Para uma amostra de n = 10.000:

| Medida | Valor |
|--------|-------|
| Total de candidatos gerados | 18.344 |
| Taxa de aceitação empírica | 54,51% |
| Taxa de aceitação teórica (1/M) | 65,77% |
| Média da amostra | −0,0043 (teórica: 0) |
| Desvio padrão da amostra | 0,9971 (teórico: 1) |
| Mediana da amostra | −0,0098 (teórica: 0) |

As estatísticas da amostra gerada (média ≈ 0, desvio padrão ≈ 1) confirmam que o método produz amostras com distribuição Normal padrão. A taxa de aceitação empírica (54,5%) ficou abaixo da teórica (65,8%) devido ao superdimensionamento do primeiro lote de candidatos. Um teste adicional foi realizado, aumentando o 'n' da função gerar_normal_aceitacao_rejeicao para 1.000.000, e os resultados foram mais próximos dos teóricos (54,81%). Desta forma, espera-se que, com amostras maiores, a taxa venha a convergir para o valor teórico.

O histograma da amostra sobreposto à densidade teórica da Normal padrão:

![Normal Padrão via Aceitação-Rejeição](grafico_normal_item_2c.png)

O gráfico do envelope ilustra como M·g(x) cobre completamente f(x), sendo a região entre as curvas a região de rejeição:

![Envelope do Método de Aceitação-Rejeição](grafico_envelope_item_2c.png)

---

## Referências

- Borges, J. L. (1941). *A Biblioteca de Babel*. Em: Ficções.
- Rizzo, M. L. (2019). *Statistical Computing with R*. 2ª ed. CRC Press.
- Wikipedia. Frequência de letras na língua portuguesa. Disponível em: [link](https://pt.wikipedia.org/wiki/Frequ%C3%AAncia_de_letras).
- Wikipedia. Cauchy distribution. Disponível em: [link](https://en.wikipedia.org/wiki/Cauchy_distribution).

---

## Anexo — Código Fonte Completo

<!-- Instrução 4: Os códigos utilizados devem ser disponibilizados na íntegra -->

### Questão 1 — `q1.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from collections import Counter
import string
import time

# =====================================================================
# DECLARAÇÃO DE USO DE LLMs (Conforme Instrução)
# =====================================================================
"""
USO DE LLMs:
A lógica para a resolução de cada um dos itens foi desenvolvida sem o auxílio do LLM.
O modelo de linguagem (Gemini) foi utilizado nesta questão para auxiliar na 
estruturação e documentação do código Python, feitas através de comentários e docstrings, 
além da otimização das rotinas de simulação (vetorização com numpy), 
visando eficiência computacional (Instrução 7), e em melhorias visuais do gráfico e dos 
resultados impressos no terminal.
"""

# =====================================================================
# PREPARAÇÃO DE DADOS: O DICIONÁRIO E AS LETRAS
# =====================================================================

def remove_acentos(texto):
    """Remove acentos de palavras e normaliza."""
    texto = unicodedata.normalize('NFD', texto)
    return ''.join(ch for ch in texto if unicodedata.category(ch) != 'Mn')

caminho_dicionario = "./Dicionario.txt"
palavras_validas = set()
letras_dicionario = []

# Lê o dicionário, processa as palavras e cria conjunto com palavras de 5 letras
with open(caminho_dicionario, "r", encoding="utf-8", errors="ignore") as f:
    for linha in f:
        palavra = linha.strip().lower()
        palavra_limpa = remove_acentos(palavra)
        
        # Filtra palavras compostas apenas de letras
        if palavra_limpa.isalpha():
            letras_dicionario.extend(list(palavra_limpa))
            if len(palavra_limpa) == 5:
                palavras_validas.add(palavra_limpa)

qtd_validas = len(palavras_validas)
alfabeto = list(string.ascii_lowercase) # 'a' a 'z'
espaco_amostral_a = 26**5

print(f"Total de palavras válidas (5 letras) no dicionário: {qtd_validas}")

# =====================================================================
# ITEM (a): Probabilidade de gerar palavra válida e convergência
# =====================================================================

ta_0 = time.time()
print("\n" + "="*40 + "\nITEM (a)\n" + "="*40)

# Probabilidade Analítica
prob_analitica_a = qtd_validas / espaco_amostral_a
print(f"Probabilidade Analítica (qtd_validas / 26^5) = {prob_analitica_a:.6f} ({prob_analitica_a*100:.4f}%)")

# Simulação Computacional via Monte Carlo
np.random.seed(42) # Semente para reprodutibilidade
N_max = 1000000
print(f"Gerando {N_max} sequências para o item (a)...")

# Geramos sequências de 5 letras de forma vetorizada para máxima eficiência
seq_indices = np.random.randint(0, 26, size=(N_max, 5))
seq_letras = np.array(alfabeto)[seq_indices]

# Concatena as letras de cada sequência e verifica validade O(1) usando o `set`
palavras_geradas = [''.join(seq) for seq in seq_letras]
print(f'Exemplos de palavras geradas: {palavras_geradas[:5]}')
validades = np.array([1 if p in palavras_validas else 0 for p in palavras_geradas])

# Estimativa pontual final
p_estimada_a = validades.sum() / N_max
print(f"Probabilidade Estimada (N={N_max}): {p_estimada_a:.6f} ({p_estimada_a*100:.4f}%)")

# Gráfico de Convergência (Calculando a proporção cumulativa)
tamanhos_amostra = np.arange(1000, N_max+1, 2000)
est_cumulativas = [np.mean(validades[:n]) for n in tamanhos_amostra]

# Estimativa teórica da margem de erro (Teorema Central do Limite)
# A "função" que rege a convergência tem decaimento proporcional a 1/sqrt(N)
Z = 1.96 # Nível de confiança de 95%
erro_padrao = np.sqrt(prob_analitica_a * (1 - prob_analitica_a) / tamanhos_amostra)
limite_superior = prob_analitica_a + Z * erro_padrao
limite_inferior = prob_analitica_a - Z * erro_padrao

plt.figure(figsize=(10, 6))
plt.plot(tamanhos_amostra, est_cumulativas, label="Estimativa (Monte Carlo)", color="blue")
plt.axhline(y=prob_analitica_a, color="red", linestyle="--", label="Valor Teórico")

# Plotando o "funil" de convergência (Intervalo de confiança)
plt.plot(tamanhos_amostra, limite_superior, color="gray", linestyle=":", label="Intervalo de Confiança (95%)")
plt.plot(tamanhos_amostra, limite_inferior, color="gray", linestyle=":")
plt.fill_between(tamanhos_amostra, limite_inferior, limite_superior, color="gray", alpha=0.2)

plt.title("Item A: Convergência da Probabilidade de Gerar Palavra Válida")
plt.xlabel("Tamanho da Amostra (N)")
plt.ylabel("Probabilidade Estimada")
plt.legend()
plt.grid(True)
plt.tight_layout()
nome_arquivo = "grafico_convergencia_item_1a.png"
plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
print(f"Gráfico de convergência salvo com sucesso em '{nome_arquivo}'.")
plt.close() # Fecha a figura para liberar memória

ta_1 = time.time()
t_a = ta_1 - ta_0
print(f"Tempo gasto no item (a): {t_a:.4f} segundos")

# =====================================================================
# ITEM (b): Probabilidade de Palíndromos
# =====================================================================

tb_0 = time.time()
print("\n" + "="*40 + "\nITEM (b)\n" + "="*40)

# Probabilidade Analítica
# Palíndromo de 5 letras: formato L1 L2 L3 L2 L1. 
# 3 graus de liberdade (L1, L2, L3) independentes de 26 opções.
# Total de casos possíveis que são palíndromos: 26^3. Espaço amostral: 26^5.
prob_analitica_b = (26**3) / (26**5) # == 1 / 26**2
print(f"Probabilidade Analítica de Palíndromo: {prob_analitica_b:.6f} ({prob_analitica_b*100:.4f}%)")

# Simulação Computacional
# Aproveitando a enorme amostra já gerada no item (a) para eficiência de recursos
# Verifica se a string é igual a ela mesma revertida
val_palindromo = np.array([1 if p == p[::-1] else 0 for p in palavras_geradas])
prob_estimada_b = val_palindromo.sum() / N_max
print(f"Probabilidade Estimada de Palíndromo (N={N_max}): {prob_estimada_b:.6f} ({prob_estimada_b*100:.4f}%)")

tb_1 = time.time()
t_b = tb_1 - tb_0
print(f"Tempo gasto no item (b): {t_b:.4f} segundos")

# =====================================================================
# ITEM (c): Gerador Alternado Consoante / Vogal
# =====================================================================

tc_0 = time.time()
print("\n" + "="*40 + "\nITEM (c)\n" + "="*40)

vogais = ['a', 'e', 'i', 'o', 'u']
consoantes = [c for c in alfabeto if c not in vogais]

N_c = 1000000
print(f"Gerando {N_c} sequências com gerador alternado...")

# Sortear a primeira letra de maneira uniforme entre as 26:
primeira_letra_char = np.random.choice(alfabeto, size=N_c)

# Criar matrizes de escolhas uniformes para as 4 posições restantes (pos. 2 a 5)
escolhas_vogais = np.random.choice(vogais, size=(N_c, 4))
escolhas_consoantes = np.random.choice(consoantes, size=(N_c, 4))

is_vogal = np.isin(primeira_letra_char, vogais)

# Montar as palavras de forma vetorizada usando numpy.char para velocidade.
# A PRIMEIRA LETRA da sequência É a que foi sorteada uniformemente do alfabeto.
# Cenário 1: Começou com vogal (Padrão: V - C - V - C - V)
vog = np.char.add(primeira_letra_char, escolhas_consoantes[:,0])
vog = np.char.add(vog, escolhas_vogais[:,0])
vog = np.char.add(vog, escolhas_consoantes[:,1])
vog = np.char.add(vog, escolhas_vogais[:,1])

# Cenário 2: Começou com consoante (Padrão: C - V - C - V - C)
cons = np.char.add(primeira_letra_char, escolhas_vogais[:,0])
cons = np.char.add(cons, escolhas_consoantes[:,0])
cons = np.char.add(cons, escolhas_vogais[:,1])
cons = np.char.add(cons, escolhas_consoantes[:,1])

# Aplica a condicional do início para juntar todas as gerações do gerador misto
geradas_c = np.where(is_vogal, vog, cons)

print(f'Exemplos de palavras geradas: {geradas_c[:5]}')

# Contar as válidas usando list comprehension simples (mais rápido para iterar strings)
validas_c_arr = [1 if p in palavras_validas else 0 for p in geradas_c]
prob_estimada_c = sum(validas_c_arr) / N_c
print(f"Probabilidade Estimada (Gerador Alternado, N={N_c}): {prob_estimada_c:.6f} ({prob_estimada_c*100:.4f}%)")

tc_1 = time.time()
t_c = tc_1 - tc_0
print(f"Tempo gasto no item (c): {t_c:.4f} segundos")


# =====================================================================
# ITEM (d): Frequência das Letras e Probabilidade Condicional
# =====================================================================

td_0 = time.time()
print("\n" + "="*40 + "\nITEM (d)\n" + "="*40)

# 1. Frequências do Dicionário
contagem_dic = Counter(letras_dicionario)
total_letras = sum(contagem_dic.values())
freqs_dicionario = {letra: contagem_dic.get(letra, 0)/total_letras for letra in alfabeto}

# 2. Frequências da Wikipedia (Língua Portuguesa)
freqs_wikipedia_pct = {
    'a': 14.63, 'b': 1.04, 'c': 3.88, 'd': 4.99, 'e': 12.57, 'f': 1.02, 'g': 1.30, 
    'h': 1.28,  'i': 6.18, 'j': 0.40, 'k': 0.02, 'l': 2.78,  'm': 4.74, 'n': 5.05, 
    'o': 10.73, 'p': 2.52, 'q': 1.20, 'r': 6.53, 's': 7.81,  't': 4.34, 'u': 4.63, 
    'v': 1.67,  'w': 0.01, 'x': 0.21, 'y': 0.01, 'z': 0.47
}
# Normalizando os percentuais da Wiki para probabilidades reais exatas (soma = 1.0)
soma_pct = sum(freqs_wikipedia_pct.values())
print(f'Percentual total normalizado - Wikipedia: {soma_pct}%')
freqs_wikipedia = {k: v/soma_pct for k, v in freqs_wikipedia_pct.items()}

print(f"--- Comparativo de Frequências (Amostra de Letras) ---")
print(f"Letra 'A': Wiki = {freqs_wikipedia['a']:.4f} | Dicionário = {freqs_dicionario['a']:.4f}")
print(f"Letra 'E': Wiki = {freqs_wikipedia['e']:.4f} | Dicionário = {freqs_dicionario['e']:.4f}")
print(f"Letra 'S': Wiki = {freqs_wikipedia['s']:.4f} | Dicionário = {freqs_dicionario['s']:.4f}")
print(f"Letra 'Z': Wiki = {freqs_wikipedia['z']:.4f} | Dicionário = {freqs_dicionario['z']:.4f}")

# Extrair lista de probabilidades na ordem correta do alfabeto
p_wiki = [freqs_wikipedia[letra] for letra in alfabeto]

# Simulação Computacional Condicional
N_d = 1000000
print(f"\nGerando {N_d} sequências usando frequências da Wikipedia...\n")

seq_indices_d = np.random.choice(26, size=(N_d, 5), p=p_wiki)
seq_letras_d = np.array(alfabeto)[seq_indices_d]
palavras_geradas_d = [''.join(seq) for seq in seq_letras_d]

# O EVENTO CONDICIONAL: A sequência ter ao menos um 'a'
tem_a = np.array(['a' in p for p in palavras_geradas_d])
palavras_com_a = np.array(palavras_geradas_d)[tem_a]

# P(Palavra Válida | Tem 'a') = Qt(Válidas com 'a') / Qt(Total gerado com 'a')
palavras_validas_geradas_com_a = [p for p in palavras_com_a if p in palavras_validas]
palavras_validas_geradas_com_a_str = [str(p) for p in palavras_validas_geradas_com_a[:5]]
qtd_validas = len(palavras_validas_geradas_com_a)
qtd_validas_unicas = len(set(palavras_validas_geradas_com_a))

prob_condicional = qtd_validas / len(palavras_com_a)
print(f"-> Total gerado que contém ao menos um 'a': {len(palavras_com_a)}")
print(f'Exemplos de palavras que contém ao menos um "a": {palavras_com_a[:5]}\n')
print(f"-> Destas, a quantidade total de palavras válidas: {qtd_validas}")
print(f'Exemplos de palavras válidas: {palavras_validas_geradas_com_a_str[:5]}\n')
print(f"-> Destas, a quantidade de palavras válidas únicas (deduplicadas): {qtd_validas_unicas}")
print(f"-> Probabilidade Condicional Estimada (Válida | contém 'a'): {prob_condicional:.6f} ({prob_condicional*100:.4f}%)")

td_1 = time.time()
t_d = td_1 - td_0
print(f"Tempo gasto no item (d): {t_d:.4f} segundos")

# =====================================================================
# RESUMO FINAL DE TEMPOS
# =====================================================================

print("=" * 60)
print("RESUMO DE TEMPOS — QUESTÃO 1")
print("=" * 60)
print(f"  Item (a):              {ta_1 - ta_0:.4f}s")
print(f"  Item (b):              {tb_1 - tb_0:.4f}s")
print(f"  Item (c):              {tc_1 - tc_0:.4f}s")
print(f"  Item (d):              {td_1 - td_0:.4f}s")
print(f"  Total:                 {td_1 - ta_0:.4f}s")
```

### Questão 2 — `q2.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Força UTF-8 no terminal Windows (evita erros com cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# DECLARAÇÃO DE USO DE LLMs (Conforme Instrução)
# =====================================================================

"""
USO DE LLMs:
A lógica para a resolução de cada um dos itens foi desenvolvida sem o auxílio do LLM.
O modelo de linguagem (Claude) foi utilizado nesta questão para auxiliar na
estruturação e documentação do código Python, feitas através de comentários e docstrings,
além de melhorias visuais dos gráficos e dos resultados impressos no terminal e auxílio
na compreensão matemática acerca da distribuição Cauchy, Transformada Inversa e Método de 
Aceitação-Rejeição.
"""

# =====================================================================
# ITEM (a): Distribuição Cauchy via Transformada Inversa
# =====================================================================

ta_0 = time.time()
print("=" * 60 + "\nITEM (a) — Cauchy via Transformada Inversa\n" + "=" * 60)


def gerar_cauchy(n, gamma, seed=None):
    """
    Gera uma amostra aleatória de tamanho n da distribuição Cauchy(gamma)
    usando o método da transformada inversa.

    Fundamentação:
        A CDF da Cauchy(gamma) é:
            F(x) = 1/2 + (1/pi) * arctan(x/gamma)

        Invertendo, obtemos:
            F^{-1}(u) = gamma * tan(pi * (u - 1/2))

        Portanto, se U ~ Uniforme(0,1), então X = gamma * tan(pi*(U - 0.5))
        segue uma distribuição Cauchy(gamma).

    Parâmetros:
        n     (int)   : tamanho da amostra desejada
        gamma (float) : parâmetro de escala da distribuição Cauchy (gamma > 0)
        seed  (int)   : semente para reprodutibilidade (opcional)

    Retorna:
        np.ndarray : vetor de tamanho n com amostras da Cauchy(gamma)
    """
    if seed is not None:
        np.random.seed(seed)

    # Gera n valores uniformes em (0, 1)
    u = np.random.uniform(0, 1, size=n)

    # Aplica a transformada inversa da CDF da Cauchy
    x = gamma * np.tan(np.pi * (u - 0.5))

    return x

# --- Demonstração com gamma = 1 (Cauchy padrão) ---
n_a = 10000
gamma_1 = 1.0
amostra_cauchy = gerar_cauchy(n=n_a, gamma=gamma_1, seed=42)

print(f"Parâmetros: n = {n_a}, gamma = {gamma_1}")
print(f"Primeiros 10 valores gerados: {np.round(amostra_cauchy[:10], 4)}")
print(f"Mediana da amostra: {np.median(amostra_cauchy):.4f} (teórica: 0)")

# --- Gráfico: Histograma vs Densidade Teórica ---
# A Cauchy tem caudas muito pesadas, então limitamos o eixo x para visualização
x_plot = np.linspace(-10, 10, 1000)
# Densidade teórica: f(x) = 1 / (pi * gamma * (1 + (x/gamma)^2))
densidade_teorica = 1.0 / (np.pi * gamma_1 * (1 + (x_plot / gamma_1) ** 2))

plt.figure(figsize=(10, 6))
# Histograma normalizado (density=True) com recorte no intervalo [-10, 10]
plt.hist(
    amostra_cauchy,
    bins=200,
    density=True,
    range=(-10, 10),
    alpha=0.6,
    color="steelblue",
    edgecolor="white",
    label=f"Histograma (n={n_a})",
)
plt.plot(
    x_plot,
    densidade_teorica,
    color="red",
    linewidth=2,
    label=f"Densidade Teórica Cauchy(γ={gamma_1})",
)
plt.title("Item 2a: Amostra Cauchy via Transformada Inversa")
plt.xlabel("x")
plt.ylabel("Densidade")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grafico_cauchy_item_2a.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo em 'grafico_cauchy_item_2a.png'.")
plt.close()

# --- Demonstração com gamma arbitrário ---
for gamma_teste in [0.5, 2.0, 5.0]:
    amostra_teste = gerar_cauchy(n=5000, gamma=gamma_teste, seed=123)
    print(
        f"  gamma={gamma_teste}: mediana={np.median(amostra_teste):.4f}, "
        f"Q1={np.percentile(amostra_teste, 25):.4f}, "
        f"Q3={np.percentile(amostra_teste, 75):.4f}"
    )

ta_1 = time.time()
print(f"Tempo gasto no item (a): {ta_1 - ta_0:.4f} segundos\n")


# =====================================================================
# ITEM (b): Distribuição Discreta via Transformada Inversa
# =====================================================================

tb_0 = time.time()
print("=" * 60 + "\nITEM (b) — Discreta via Transformada Inversa\n" + "=" * 60)

# Função massa de probabilidade fornecida no enunciado
mp_x = np.array([2, 3, 5, 6, 9])
prob = np.array([0.3, 0.1, 0.1, 0.3, 0.2])

# CDF acumulada: F(2)=0.3, F(3)=0.4, F(5)=0.5, F(6)=0.8, F(9)=1.0
cdf_acumulada = np.cumsum(prob)
print(f"Valores de X:      {mp_x}")
print(f"Probabilidades:    {prob}")
print(f"CDF acumulada:     {cdf_acumulada}")


def gerar_discreta_inversa(n, mp_x, cdf, seed=None):
    """
    Gera uma amostra aleatória de tamanho n de uma distribuição discreta
    usando o método da transformada inversa.

    Fundamentação:
        Dado U ~ Uniforme(0,1), atribui-se X = x_i onde i é o menor índice
        tal que U <= F(x_i), sendo F a CDF acumulada.

        Exemplo com a PMF do enunciado:
            U em (0.0, 0.3] -> X = 2
            U em (0.3, 0.4] -> X = 3
            U em (0.4, 0.5] -> X = 5
            U em (0.5, 0.8] -> X = 6
            U em (0.8, 1.0] -> X = 9

    Parâmetros:
        n       (int)        : tamanho da amostra
        mp_x    (np.ndarray) : valores possíveis de X (ordenados)
        cdf     (np.ndarray) : CDF acumulada correspondente
        seed    (int)        : semente para reprodutibilidade (opcional)

    Retorna:
        np.ndarray : vetor de tamanho n com amostras da distribuição discreta
    """
    if seed is not None:
        np.random.seed(seed)

    u = np.random.uniform(0, 1, size=n)

    # np.searchsorted encontra, para cada u, o menor índice i tal que cdf[i] >= u
    # Isso implementa a transformada inversa discreta de forma vetorizada
    indices = np.searchsorted(cdf, u, side="left")

    return mp_x[indices]


# --- Método 1: Transformada Inversa ---
n_b = 1000
amostra_inversa = gerar_discreta_inversa(n=n_b, mp_x=mp_x, cdf=cdf_acumulada, seed=42)

# Calcula frequências relativas
freq_inversa = {val: np.sum(amostra_inversa == val) / n_b for val in mp_x}

# --- Método 2: np.random.choice (similar ao sample do R) ---
# Usa seed diferente para demonstrar que ambos convergem independentemente
np.random.seed(123)
amostra_choice = np.random.choice(mp_x, size=n_b, p=prob)
freq_choice = {val: np.sum(amostra_choice == val) / n_b for val in mp_x}

# --- Tabela Comparativa ---
print(f"\n{'─' * 70}")
print(f"{'x':^6} | {'P(x) Teórica':^14} | {'Transf. Inversa':^17} | {'np.random.choice':^18}")
print(f"{'─' * 70}")
for val, p_teo in zip(mp_x, prob):
    fi = freq_inversa[val]
    fc = freq_choice[val]
    print(f"{val:^6} | {p_teo:^14.2f} | {fi:^17.4f} | {fc:^18.4f}")
print(f"{'─' * 70}")

# --- Gráfico comparativo ---
largura_barra = 0.25
posicoes = np.arange(len(mp_x))

plt.figure(figsize=(10, 6))
plt.bar(
    posicoes - largura_barra,
    prob,
    largura_barra,
    label="P(x) Teórica",
    color="gray",
    edgecolor="black",
    alpha=0.8,
)
plt.bar(
    posicoes,
    [freq_inversa[v] for v in mp_x],
    largura_barra,
    label="Transf. Inversa",
    color="steelblue",
    edgecolor="black",
    alpha=0.8,
)
plt.bar(
    posicoes + largura_barra,
    [freq_choice[v] for v in mp_x],
    largura_barra,
    label="np.random.choice",
    color="coral",
    edgecolor="black",
    alpha=0.8,
)

plt.xlabel("x")
plt.ylabel("Frequência Relativa / Probabilidade")
plt.title("Item 2b: Comparação das Frequências Empíricas com as Teóricas")
plt.xticks(posicoes, mp_x)
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("grafico_discreta_item_2b.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo em 'grafico_discreta_item_2b.png'.")
plt.close()

tb_1 = time.time()
print(f"Tempo gasto no item (b): {tb_1 - tb_0:.4f} segundos\n")


# =====================================================================
# ITEM (c): Normal Padrão via Aceitação e Rejeição (Cauchy como geradora)
# =====================================================================

tc_0 = time.time()
print("=" * 60 + "\nITEM (c) — Normal via Aceitação-Rejeição\n" + "=" * 60)

"""
Fundamentação Teórica:

O método de aceitação e rejeição (acceptance-rejection) permite gerar amostras
de uma distribuição-alvo f(x) usando uma distribuição geradora (proposta) g(x),
desde que exista uma constante M >= 1 tal que f(x) <= M * g(x) para todo x.

  - Distribuição-alvo:   f(x) = (1/sqrt(2*pi)) * exp(-x^2/2)  (Normal padrão)
  - Distribuição geradora: g(x) = 1 / (pi * (1 + x^2))         (Cauchy padrão, gamma=1)

  - Constante envolvente M:
      M = sup_x [f(x) / g(x)]
        = sup_x [(1/sqrt(2*pi)) * exp(-x^2/2) * pi * (1 + x^2)]
        = (pi / sqrt(2*pi)) * sup_x [(1 + x^2) * exp(-x^2/2)]

      Derivando h(x) = (1 + x^2) * exp(-x^2/2) e igualando a zero:
        h'(x) = x * exp(-x^2/2) * (1 - x^2) = 0  =>  x = 0 (máximo, pois h''(0) < 0 e x=±1 é mínimo local de h'/h)
        Nota: na verdade, h'(x) = [2x - x(1+x^2)] * exp(-x^2/2) = x(1-x^2)*exp(-x^2/2)
        h'(x) = 0 em x=0, x=1, x=-1. h(0)=1, h(1)=2/sqrt(e)≈1.2131, h(-1)=2/sqrt(e).
        O máximo é em x=±1: h(1) = 2*exp(-1/2) = 2/sqrt(e)

      Portanto:
        M = (pi / sqrt(2*pi)) * 2/sqrt(e) = sqrt(2*pi/e) ≈ 1.5203

Algoritmo:
    1. Gerar Y ~ Cauchy(1) usando a função do item (a)
    2. Gerar U ~ Uniforme(0, 1)
    3. Calcular a razão de aceitação: r = f(Y) / (M * g(Y))
    4. Se U <= r, aceitar Y como amostra da Normal; caso contrário, rejeitar e repetir

A taxa de aceitação teórica é 1/M ≈ 0.6577 (≈ 65.8%).
"""


def gerar_normal_aceitacao_rejeicao(n, seed=None):
    """
    Gera uma amostra de tamanho n da distribuição Normal padrão (mu=0, sigma=1)
    usando o método de aceitação-rejeição com a Cauchy padrão (gamma=1) como
    distribuição geradora de candidatos.

    Parâmetros:
        n    (int) : tamanho da amostra desejada
        seed (int) : semente para reprodutibilidade (opcional)

    Retorna:
        tuple: (amostra, total_gerados, taxa_aceitacao)
            - amostra       (np.ndarray) : vetor de tamanho n com amostras da Normal(0,1)
            - total_gerados (int)         : número total de candidatos gerados
            - taxa_aceitacao (float)      : fração de candidatos aceitos (n / total_gerados)
    """
    if seed is not None:
        np.random.seed(seed)

    # Constante envolvente M = sqrt(2*pi/e)
    M = np.sqrt(2 * np.pi / np.e)

    amostra = []
    gerados = 0

    while len(amostra) < n:
        # Gera candidatos em lote para eficiência (evita loop unitário lento)
        # O tamanho do lote é estimado com base na taxa de aceitação teórica (1/M)
        # mais uma margem de segurança de 20%
        lote_size = int((n - len(amostra)) * M * 1.2) + 100

        # Passo 1: Gerar candidatos Y ~ Cauchy(1) via transformada inversa
        y = gerar_cauchy(n=lote_size, gamma=1.0)

        # Passo 2: Gerar U ~ Uniforme(0, 1)
        u = np.random.uniform(0, 1, size=lote_size)

        # Passo 3: Calcular a razão de aceitação f(y) / (M * g(y))
        # f(y) = (1/sqrt(2*pi)) * exp(-y^2/2)           (Normal padrão)
        # g(y) = 1 / (pi * (1 + y^2))                   (Cauchy padrão)
        # M * g(y) = sqrt(2*pi/e) / (pi * (1 + y^2))
        #
        # Simplificando f(y) / (M * g(y)):
        #   = [(1/sqrt(2*pi)) * exp(-y^2/2)] / [sqrt(2*pi/e) / (pi * (1 + y^2))]
        #   = [(1/sqrt(2*pi)) * exp(-y^2/2) * pi * (1 + y^2)] / sqrt(2*pi/e)
        #   = (1 + y^2) * exp(-y^2/2) * sqrt(e) / 2
        #   = (1 + y^2) * exp(-(y^2 - 1)/2) / 2
        razao = (1 + y**2) * np.exp(-(y**2 - 1) / 2) / 2

        # Passo 4: Aceitar se U <= razão
        aceitos = y[u <= razao]
        amostra.extend(aceitos.tolist())
        gerados += lote_size

    # Recorta para exatamente n amostras (pode ter gerado um pouco mais)
    amostra = np.array(amostra[:n])
    taxa_aceitacao = n / gerados

    return amostra, gerados, taxa_aceitacao


# --- Geração da amostra Normal ---
n_c = 10000
amostra_normal, gerados_c, taxa_c = gerar_normal_aceitacao_rejeicao(
    n=n_c, seed=42
)

# --- Estatísticas descritivas ---
M_teorico = np.sqrt(2 * np.pi / np.e)
taxa_teorica = 1 / M_teorico

print(f"Parâmetros: n = {n_c}")
print(f"Total de candidatos gerados: {gerados_c}")
print(f"Taxa de aceitação empírica:  {taxa_c:.4f} ({taxa_c*100:.2f}%)")
print(f"Taxa de aceitação teórica:   {taxa_teorica:.4f} ({taxa_teorica*100:.2f}%)")
print(f"\nEstatísticas da amostra gerada:")
print(f"  Média:         {np.mean(amostra_normal):.4f}  (teórica: 0)")
print(f"  Desvio padrão: {np.std(amostra_normal):.4f}   (teórico: 1)")
print(f"  Mediana:       {np.median(amostra_normal):.4f} (teórica: 0)")

# --- Gráfico: Histograma da amostra vs Densidade Normal teórica ---
x_plot = np.linspace(-5, 5, 1000)
# Densidade da Normal padrão: f(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
normal_teorica = (1 / np.sqrt(2 * np.pi)) * np.exp(-(x_plot**2) / 2)

plt.figure(figsize=(10, 6))
plt.hist(
    amostra_normal,
    bins=80,
    density=True,
    alpha=0.6,
    color="steelblue",
    edgecolor="white",
    label=f"Histograma (n={n_c})",
)
plt.plot(
    x_plot,
    normal_teorica,
    color="red",
    linewidth=2,
    label="Densidade Teórica N(0,1)",
)
plt.title("Item 2c: Amostra Normal Padrão via Aceitação-Rejeição")
plt.xlabel("x")
plt.ylabel("Densidade")
plt.legend()
plt.grid(True, alpha=0.3)

# Anotação com a taxa de aceitação no gráfico
plt.annotate(
    f"Taxa de aceitação: {taxa_c*100:.1f}%\n(teórica: {taxa_teorica*100:.1f}%)",
    xy=(0.97, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"),
)

plt.tight_layout()
plt.savefig("grafico_normal_item_2c.png", dpi=300, bbox_inches="tight")
print("Gráfico salvo em 'grafico_normal_item_2c.png'.")
plt.close()

# --- Gráfico adicional: Envelope M*g(x) vs f(x) ---
# Visualização didática de como a envolvente M*g(x) cobre f(x)
cauchy_teorica = 1 / (np.pi * (1 + x_plot**2))
envelope = M_teorico * cauchy_teorica

plt.figure(figsize=(10, 6))
plt.plot(x_plot, normal_teorica, color="blue", linewidth=2, label="f(x) — Normal(0,1)")
plt.plot(
    x_plot,
    envelope,
    color="red",
    linewidth=2,
    linestyle="--",
    label=f"M·g(x) — {M_teorico:.4f} × Cauchy(1)",
)
plt.fill_between(x_plot, normal_teorica, envelope, alpha=0.15, color="red",
                 label="Região de rejeição")
plt.fill_between(x_plot, 0, normal_teorica, alpha=0.15, color="blue",
                 label="Região de aceitação")
plt.title("Item 2c: Envelope do Método de Aceitação-Rejeição")
plt.xlabel("x")
plt.ylabel("Densidade")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grafico_envelope_item_2c.png", dpi=300, bbox_inches="tight")
print("Gráfico do envelope salvo em 'grafico_envelope_item_2c.png'.")
plt.close()

tc_1 = time.time()
print(f"Tempo gasto no item (c): {tc_1 - tc_0:.4f} segundos\n")

# =====================================================================
# RESUMO FINAL DE TEMPOS
# =====================================================================

print("=" * 60)
print("RESUMO DE TEMPOS — QUESTÃO 2")
print("=" * 60)
print(f"  Item (a) - Cauchy:              {ta_1 - ta_0:.4f}s")
print(f"  Item (b) - Discreta:            {tb_1 - tb_0:.4f}s")
print(f"  Item (c) - Normal (Aceit/Rej):  {tc_1 - tc_0:.4f}s")
print(f"  Total:                          {tc_1 - ta_0:.4f}s")
```
