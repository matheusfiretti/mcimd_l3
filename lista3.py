"""
Lista 3 - Métodos Computacionais Intensivos para Mineração de Dados
Universidade de Brasília - Departamento de Ciência da Computação
Prof. Guilherme Rodrigues | 1/2026

Aluno: Matheus Firetti

Tema: Inferência Bayesiana para parâmetros de uma fila M/M/2
usando Approximate Bayesian Computation (ABC).

Referências:
  - Ebert et al. (2019), "Computationally Efficient Simulation of Queues:
    The R Package queuecomputer" (arxiv: 1703.02151)
  - Rodrigues, G. - Tese de Doutorado (Cap. 1: Background on ABC)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
import sys
import concurrent.futures
import multiprocessing

# Forçar UTF-8 no stdout (necessário no Windows com terminal cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# DECLARAÇÃO DE USO DE LLMs (Conforme Instrução)
# =====================================================================
"""
USO DE LLMs:
O modelo de linguagem (Claude) foi utilizado para auxiliar na estruturação
e documentação do código Python, por meio de comentários e docstrings,
e na explicação dos conceitos de ABC e teoria de filas presentes nos
comentários ao longo do código. A lógica de resolução de cada questão
foi desenvolvida com base no material de estudo indicado.
"""

# =====================================================================
# CONFIGURAÇÕES GERAIS
# =====================================================================

SEMENTE = 42
np.random.seed(SEMENTE)

# --- Parâmetros do problema ---
TEMPO_MAX = 600       # minutos (8h-18h = 10 horas = 600 min)
N_SERV = 2            # fila M/M/2 (dois servidores)
W_OBS = 150.0         # tempo médio de espera observado (2h30 = 150 min)
LQ_OBS = 22.0         # comprimento médio da fila (apenas quem espera)

# --- Limites das prioris uniformes ---
LAM_MIN, LAM_MAX = 0.01, 2.0      # λ: taxa de chegada (clientes/min)
MU_MIN, MU_MAX = 0.005, 1.0       # μ: taxa de atendimento (atend./min)

# (parâmetros do ABC definidos por estágio na Q1)


# =====================================================================
# SIMULADOR DE FILA M/M/K — ALGORITMO QDC
# =====================================================================

def simular_fila(lam, mu, tempo_max=TEMPO_MAX, n_serv=N_SERV):
    """
    Simula uma fila M/M/K via Queue Departure Computation (QDC).

    Referência: Ebert et al. (2019), Algorithm 1.

    O QDC "desacopla" a geração de amostras do cálculo da fila:
    geramos todas as chegadas e tempos de serviço de antemão,
    depois calculamos as partidas deterministicamente usando um vetor
    b[k] que rastreia quando cada servidor k fica disponível.

    Parâmetros:
        lam       : taxa de chegada (clientes/min), parâmetro da Exp
        mu        : taxa de atendimento por servidor (atend./min)
        tempo_max : janela de observação em minutos
        n_serv    : número de servidores (K)

    Retorna:
        dict com: chegadas, servicos, partidas, esperas,
                  servidores (alocação), n_clientes
    """
    # O nº esperado de chegadas é λ*T; geramos ~2x pra ter margem,
    # com piso de 200 (o sample em R da lista usava N fixo = 200,
    # mas aqui variamos dinamicamente p/ cobrir o intervalo).
    n_gen = max(int(lam * tempo_max * 2) + 100, 200)
    interchegadas = np.random.exponential(1.0 / lam, size=n_gen)
    chegadas = np.cumsum(interchegadas)
    chegadas = chegadas[chegadas <= tempo_max]

    n_clientes = len(chegadas)
    if n_clientes == 0:
        return {
            'chegadas': np.array([]), 'servicos': np.array([]),
            'partidas': np.array([]), 'esperas': np.array([]),
            'servidores': np.array([], dtype=int), 'n_clientes': 0
        }

    servicos = np.random.exponential(1.0 / mu, size=n_clientes)

    # Vetor b: quando cada servidor fica disponível (init = 0)
    b = np.zeros(n_serv)
    esperas = np.zeros(n_clientes)
    partidas = np.zeros(n_clientes)
    srv_alocado = np.zeros(n_clientes, dtype=int)

    for i in range(n_clientes):
        # Escolhe o servidor que fica livre mais cedo
        srv = int(np.argmin(b))

        # Se o servidor ainda está ocupado quando o cliente chega, espera
        espera_i = max(0.0, b[srv] - chegadas[i])
        b[srv] = max(chegadas[i], b[srv]) + servicos[i]

        esperas[i] = espera_i
        partidas[i] = b[srv]
        srv_alocado[i] = srv

    return {
        'chegadas': chegadas, 'servicos': servicos,
        'partidas': partidas, 'esperas': esperas,
        'servidores': srv_alocado, 'n_clientes': n_clientes
    }


def calcular_lq_medio(chegadas, partidas, tempo_max, n_serv):
    """
    Calcula o Lq médio (comprimento médio da fila, apenas quem espera)
    ao longo do intervalo [0, tempo_max].

    Usa integração por eventos: a cada chegada ou partida, o número
    de clientes no sistema muda. O Lq(t) = max(0, N(t) - K) é constante
    entre eventos consecutivos, então acumulamos Lq(t) * Δt.

    Lq conta APENAS quem está esperando na fila, sem incluir
    os clientes que já estão sendo atendidos.
    """
    n = len(chegadas)
    if n == 0:
        return 0.0

    # Criar timeline de eventos: +1 para chegada, -1 para partida
    tempos_ev = np.concatenate([chegadas, partidas])
    tipos_ev = np.concatenate([np.ones(n), -np.ones(n)])

    # Ordenar cronologicamente
    ordem = np.argsort(tempos_ev, kind='mergesort')
    tempos_ev = tempos_ev[ordem]
    tipos_ev = tipos_ev[ordem]

    n_sistema = 0
    lq_acum = 0.0
    t_ant = 0.0

    for j in range(len(tempos_ev)):
        t = tempos_ev[j]
        if t > tempo_max:
            break

        # Acumular Lq do intervalo anterior
        lq = max(0, n_sistema - n_serv)
        lq_acum += lq * (t - t_ant)
        t_ant = t
        n_sistema += int(tipos_ev[j])

    # Trecho restante até tempo_max
    if t_ant < tempo_max:
        lq = max(0, n_sistema - n_serv)
        lq_acum += lq * (tempo_max - t_ant)

    return lq_acum / tempo_max


def calcular_quantil_lq(chegadas, partidas, tempo_max, n_serv, quantil=0.90):
    """
    Encontra o menor C tal que Lq(t) <= C durante pelo menos
    'quantil' fração do tempo total.

    Usado na Q3 para determinar o número de cadeiras necessárias.
    Percorre a timeline de eventos e acumula tempo gasto em cada
    nível de Lq, depois varre os níveis em ordem crescente até
    atingir a fração cumulativa desejada.
    """
    n = len(chegadas)
    if n == 0:
        return 0

    tempos_ev = np.concatenate([chegadas, partidas])
    tipos_ev = np.concatenate([np.ones(n), -np.ones(n)])
    ordem = np.argsort(tempos_ev, kind='mergesort')
    tempos_ev = tempos_ev[ordem]
    tipos_ev = tipos_ev[ordem]

    tempo_por_nivel = {}
    n_sistema = 0
    t_ant = 0.0

    for j in range(len(tempos_ev)):
        t = tempos_ev[j]
        if t > tempo_max:
            break

        lq = max(0, n_sistema - n_serv)
        dt = t - t_ant
        if dt > 0:
            tempo_por_nivel[lq] = tempo_por_nivel.get(lq, 0.0) + dt
        t_ant = t
        n_sistema += int(tipos_ev[j])

    # Trecho final
    if t_ant < tempo_max:
        lq = max(0, n_sistema - n_serv)
        dt = tempo_max - t_ant
        tempo_por_nivel[lq] = tempo_por_nivel.get(lq, 0.0) + dt

    # Varre níveis crescentes até atingir o quantil
    niveis = sorted(tempo_por_nivel.keys())
    acumulado = 0.0
    for c in niveis:
        acumulado += tempo_por_nivel[c] / tempo_max
        if acumulado >= quantil:
            return int(c)

    return int(niveis[-1]) if niveis else 0


def calcular_distancia(w_sim, lq_sim, w_obs=W_OBS, lq_obs=LQ_OBS):
    """
    Distância euclidiana normalizada por erro relativo.
    Garante que ambas as estatísticas (com escalas diferentes)
    contribuam igualmente para a distância.
    """
    dw = (w_sim - w_obs) / w_obs
    dlq = (lq_sim - lq_obs) / lq_obs
    return np.sqrt(dw**2 + dlq**2)


# =====================================================================
# FUNÇÕES PARA QUESTÃO 5 — BALKING (DESISTÊNCIA)
# =====================================================================

def prob_desistencia(lq, alpha, k0):
    """
    Probabilidade de desistência via função logística.
    Modela transição suave de "fico" para "desisto" conforme Lq cresce.

    p(desistir | Lq = k) = 1 / (1 + exp(-α(k - k₀)))

    Parâmetros:
        lq    : número de pessoas na fila (apenas esperando)
        alpha : sensibilidade (quão rápido p sobe)
        k0    : limiar de conforto (Lq onde p = 50%)
    """
    return 1.0 / (1.0 + np.exp(-alpha * (lq - k0)))


def simular_fila_balking(lam, mu, tempo_max, n_serv, alpha, k0):
    """
    Simula M/M/K com desistência (balking).

    Diferente do QDC puro, aqui precisamos verificar o estado da fila
    a cada chegada para decidir se o cliente entra ou desiste.
    Isso torna o loop sequencial obrigatório (não vetorizável).

    A cada chegada potencial:
      1. Calcula Lq (pessoas esperando) naquele instante
      2. Se Lq > 0, aplica a probabilidade de desistência
      3. Se desiste, o cliente é contado mas não entra na fila
    """
    n_gen = max(int(lam * tempo_max * 2) + 100, 200)
    interchegadas = np.random.exponential(1.0 / lam, size=n_gen)
    todas_chegadas = np.cumsum(interchegadas)
    todas_chegadas = todas_chegadas[todas_chegadas <= tempo_max]

    b = np.zeros(n_serv)

    chegadas_efetivas = []
    esperas_lista = []
    partidas_lista = []
    n_desistencias = 0

    for i in range(len(todas_chegadas)):
        t_cheg = todas_chegadas[i]

        # Quantos já entraram e quantos já saíram até agora
        n_dentro = len(chegadas_efetivas)
        n_saiu = sum(1 for d in partidas_lista if d <= t_cheg)
        n_sistema = n_dentro - n_saiu
        lq_agora = max(0, n_sistema - n_serv)

        # Decisão de balking (só se houver fila)
        if lq_agora > 0:
            p_balk = prob_desistencia(lq_agora, alpha, k0)
            if np.random.random() < p_balk:
                n_desistencias += 1
                continue

        # Cliente entra na fila normalmente
        srv = int(np.argmin(b))
        espera = max(0.0, b[srv] - t_cheg)
        t_servico = np.random.exponential(1.0 / mu)
        b[srv] = max(t_cheg, b[srv]) + t_servico

        chegadas_efetivas.append(t_cheg)
        esperas_lista.append(espera)
        partidas_lista.append(b[srv])

    return {
        'chegadas': np.array(chegadas_efetivas),
        'esperas': np.array(esperas_lista),
        'partidas': np.array(partidas_lista),
        'n_clientes': len(chegadas_efetivas),
        'desistencias': n_desistencias,
        'potenciais': len(todas_chegadas)
    }



# =====================================================================
# WORKERS PARA PARALELIZAÇÃO (MULTIPROCESSING)
# =====================================================================

def _worker_abc_chunk(args):
    lam_chunk, mu_chunk, tempo_max, n_serv = args
    resultados = []
    for lam, mu in zip(lam_chunk, mu_chunk):
        res = simular_fila(lam, mu, tempo_max=tempo_max, n_serv=n_serv)
        if res['n_clientes'] < 3:
            resultados.append((np.inf, np.inf, np.inf))
            continue
        w_i = np.mean(res['esperas'])
        lq_i = calcular_lq_medio(res['chegadas'], res['partidas'], tempo_max, n_serv)
        dist = calcular_distancia(w_i, lq_i)
        resultados.append((w_i, lq_i, dist))
    return resultados

def _worker_q3(args):
    lam, mu, tempo_max, n_serv = args
    res_q3 = simular_fila(lam, mu, tempo_max=tempo_max, n_serv=n_serv)
    if res_q3['n_clientes'] < 2:
        return 0
    return calcular_quantil_lq(res_q3['chegadas'], res_q3['partidas'], tempo_max, n_serv, 0.90)

def _worker_q4(args):
    lam, mu, n_serv_a, n_serv_b = args
    res_a = simular_fila(lam, mu, n_serv=n_serv_a)
    w_mm2 = np.mean(res_a['esperas']) if res_a['n_clientes'] > 0 else 0
    res_b1 = simular_fila(lam / 2, mu, n_serv=n_serv_b)
    res_b2 = simular_fila(lam / 2, mu, n_serv=n_serv_b)
    total_b = res_b1['n_clientes'] + res_b2['n_clientes']
    w_2mm1 = (np.sum(res_b1['esperas']) + np.sum(res_b2['esperas'])) / total_b if total_b > 0 else 0
    return w_mm2, w_2mm1

def _worker_abc_balking_chunk(args):
    lam_chunk, mu_chunk, tempo_max, n_serv, alpha, k0 = args
    resultados = []
    for lam, mu in zip(lam_chunk, mu_chunk):
        res = simular_fila_balking(lam, mu, tempo_max=tempo_max, n_serv=n_serv, alpha=alpha, k0=k0)
        if res['n_clientes'] < 3:
            resultados.append((np.inf, np.inf, np.inf, 0, 0))
            continue
        w_i = np.mean(res['esperas'])
        lq_i = calcular_lq_medio(res['chegadas'], res['partidas'], tempo_max, n_serv)
        dist = calcular_distancia(w_i, lq_i)
        resultados.append((w_i, lq_i, dist, res['desistencias'], res['potenciais']))
    return resultados

def executar_abc_balking(n_amostras, lam_range, mu_range, frac_aceite, alpha, k0, rotulo=""):
    lam_lo, lam_hi = lam_range
    mu_lo, mu_hi = mu_range

    lam_sorteio = np.random.uniform(lam_lo, lam_hi, size=n_amostras)
    mu_sorteio = np.random.uniform(mu_lo, mu_hi, size=n_amostras)

    w_sims = np.zeros(n_amostras)
    lq_sims = np.zeros(n_amostras)
    dists = np.full(n_amostras, np.inf)
    desists = np.zeros(n_amostras, dtype=int)
    potenciais = np.zeros(n_amostras, dtype=int)

    t0 = time.time()
    
    n_workers = multiprocessing.cpu_count()
    chunk_size = max(1, n_amostras // n_workers)
    
    chunks = []
    for i in range(0, n_amostras, chunk_size):
        lam_c = lam_sorteio[i:i+chunk_size]
        mu_c = mu_sorteio[i:i+chunk_size]
        chunks.append((lam_c, mu_c, TEMPO_MAX, N_SERV, alpha, k0))
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_chunks = list(executor.map(_worker_abc_balking_chunk, chunks))
        
    idx = 0
    for chunk_res in results_chunks:
        for (w_i, lq_i, dist, d, p) in chunk_res:
            w_sims[idx] = w_i
            lq_sims[idx] = lq_i
            dists[idx] = dist
            desists[idx] = d
            potenciais[idx] = p
            idx += 1
            if idx % 10000 == 0:
                elapsed = time.time() - t0
                pct = 100 * idx / n_amostras
                print(f"    [{pct:5.1f}%] {idx:,}/{n_amostras:,} ({elapsed:.1f}s)")

    n_aceitos = max(int(n_amostras * frac_aceite), 50)
    ordem = np.argsort(dists)
    aceitos = ordem[:n_aceitos]
    eps = dists[aceitos[-1]]

    tempo_total = time.time() - t0
    if rotulo:
        print(f"    {rotulo}: {n_aceitos} aceitas, ε = {eps:.4f}, tempo = {tempo_total:.1f}s")

    return {
        'lam': lam_sorteio[aceitos], 'mu': mu_sorteio[aceitos],
        'w': w_sims[aceitos], 'lq': lq_sims[aceitos],
        'dist': dists[aceitos], 'eps': eps,
        'desistencias': desists[aceitos],
        'potenciais': potenciais[aceitos],
        'n_aceitos': n_aceitos, 'tempo': tempo_total
    }

# =====================================================================
# FUNÇÃO ABC REUTILIZÁVEL
# =====================================================================

def executar_abc(n_amostras, lam_range, mu_range, frac_aceite, rotulo=""):
    """
    ABC rejection sampling para (λ, μ) de uma fila M/M/2.

    Sorteia n_amostras pares (λ, μ) das prioris uniformes, simula uma fila
    para cada par, calcula estatísticas resumo (w̄, Lq) e retém os top
    'frac_aceite' com menor distância ao observado.
    """
    lam_lo, lam_hi = lam_range
    mu_lo, mu_hi = mu_range

    lam_sorteio = np.random.uniform(lam_lo, lam_hi, size=n_amostras)
    mu_sorteio = np.random.uniform(mu_lo, mu_hi, size=n_amostras)

    w_sims = np.zeros(n_amostras)
    lq_sims = np.zeros(n_amostras)
    dists = np.full(n_amostras, np.inf)

    t0 = time.time()
    
    n_workers = multiprocessing.cpu_count()
    chunk_size = max(1, n_amostras // n_workers)
    
    chunks = []
    for i in range(0, n_amostras, chunk_size):
        lam_c = lam_sorteio[i:i+chunk_size]
        mu_c = mu_sorteio[i:i+chunk_size]
        chunks.append((lam_c, mu_c, TEMPO_MAX, N_SERV))
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_chunks = list(executor.map(_worker_abc_chunk, chunks))
        
    idx = 0
    for c_idx, chunk_res in enumerate(results_chunks):
        for (w_i, lq_i, dist) in chunk_res:
            w_sims[idx] = w_i
            lq_sims[idx] = lq_i
            dists[idx] = dist
            idx += 1
            if idx % 25000 == 0:
                elapsed = time.time() - t0
                pct = 100 * idx / n_amostras
                print(f"    [{pct:5.1f}%] {idx:,}/{n_amostras:,} ({elapsed:.1f}s)")

    # Reter os melhores (menor distância)
    n_aceitos = max(int(n_amostras * frac_aceite), 50)
    ordem = np.argsort(dists)
    aceitos = ordem[:n_aceitos]
    eps = dists[aceitos[-1]]

    tempo_total = time.time() - t0
    if rotulo:
        print(f"    {rotulo}: {n_aceitos} aceitas, ε = {eps:.4f}, "
              f"tempo = {tempo_total:.1f}s")

    return {
        'lam': lam_sorteio[aceitos], 'mu': mu_sorteio[aceitos],
        'w': w_sims[aceitos], 'lq': lq_sims[aceitos],
        'dist': dists[aceitos], 'eps': eps,
        'n_aceitos': n_aceitos, 'tempo': tempo_total
    }


def main():
    # =====================================================================
    # QUESTÃO 1 — ESTIMAÇÃO DA POSTERIORI VIA ABC (3 ESTÁGIOS)
    # =====================================================================

    print("=" * 60)
    print("QUESTÃO 1 — Estimação da Posteriori de (λ, μ) via ABC")
    print("=" * 60)

    t0_q1 = time.time()

    print(f"Estatísticas observadas: w̄ = {W_OBS} min, L̄q = {LQ_OBS} pessoas")
    print(f"Estratégia: ABC rejection em 3 estágios (refinamento recursivo)\n")

    # Configuração por estágio.
    # Prof sugeriu ajustar nº de amostras e taxa de aceitação "recursivamente",
    # então cada estágio estreita a região e aumenta o rigor.
    estagios = [
        {'n': 100_000, 'pct': 0.05,  'margem': 0.15},  # amplo
        {'n': 150_000, 'pct': 0.02,  'margem': 0.10},  # refinamento
        {'n': 200_000, 'pct': 0.005, 'margem': None},   # final
    ]

    lam_lim = (LAM_MIN, LAM_MAX)
    mu_lim = (MU_MIN, MU_MAX)
    hist_eps = []

    for k, stg in enumerate(estagios):
        num = k + 1
        print(f"  Estágio {num}: n = {stg['n']:,}, "
              f"aceitação = {stg['pct']*100:.1f}%")
        print(f"    λ ∈ [{lam_lim[0]:.4f}, {lam_lim[1]:.4f}]")
        print(f"    μ ∈ [{mu_lim[0]:.4f}, {mu_lim[1]:.4f}]")

        res_abc = executar_abc(
            n_amostras=stg['n'],
            lam_range=lam_lim,
            mu_range=mu_lim,
            frac_aceite=stg['pct'],
            rotulo=f"Estágio {num}"
        )
        hist_eps.append(res_abc['eps'])

        # Estreitar prioris para o próximo estágio
        if stg['margem'] is not None:
            m = stg['margem']
            lam_lim = (
                max(LAM_MIN, np.quantile(res_abc['lam'], 0.05) * (1 - m)),
                min(LAM_MAX, np.quantile(res_abc['lam'], 0.95) * (1 + m))
            )
            mu_lim = (
                max(MU_MIN, np.quantile(res_abc['mu'], 0.05) * (1 - m)),
                min(MU_MAX, np.quantile(res_abc['mu'], 0.95) * (1 + m))
            )
            print(f"    → Prioris p/ estágio {num+1}: "
                  f"λ ∈ [{lam_lim[0]:.4f}, {lam_lim[1]:.4f}], "
                  f"μ ∈ [{mu_lim[0]:.4f}, {mu_lim[1]:.4f}]")
        print()

    # Evolução do ε
    print("  Evolução do ε:")
    for k, ep in enumerate(hist_eps):
        print(f"    Estágio {k+1}: ε = {ep:.4f}")

    # ---- Posterior final ----
    lam_post = res_abc['lam']
    mu_post = res_abc['mu']
    w_post = res_abc['w']
    lq_post = res_abc['lq']
    dist_post = res_abc['dist']
    eps_final = res_abc['eps']
    rho_post = lam_post / (2 * mu_post)

    t1_q1 = time.time()

    print(f"\n--- Resultados Finais (Estágio 3) ---")
    print(f"  Amostras aceitas: {res_abc['n_aceitos']}")
    print(f"  ε final: {eps_final:.4f}")
    print(f"\n  λ (taxa de chegada, clientes/min):")
    print(f"    Média = {np.mean(lam_post):.4f}  |  "
          f"Mediana = {np.median(lam_post):.4f}")
    print(f"    IC 95%: [{np.quantile(lam_post, 0.025):.4f}, "
          f"{np.quantile(lam_post, 0.975):.4f}]")
    print(f"    → Tempo médio entre chegadas: ~{1/np.median(lam_post):.1f} min")
    print(f"\n  μ (taxa de atendimento, atend./min):")
    print(f"    Média = {np.mean(mu_post):.4f}  |  "
          f"Mediana = {np.median(mu_post):.4f}")
    print(f"    IC 95%: [{np.quantile(mu_post, 0.025):.4f}, "
          f"{np.quantile(mu_post, 0.975):.4f}]")
    print(f"    → Tempo médio de atendimento: ~{1/np.median(mu_post):.1f} min")
    print(f"\n  ρ = λ/(2μ) (utilização do sistema):")
    print(f"    Média = {np.mean(rho_post):.4f}  |  "
          f"Mediana = {np.median(rho_post):.4f}")
    print(f"    IC 95%: [{np.quantile(rho_post, 0.025):.4f}, "
          f"{np.quantile(rho_post, 0.975):.4f}]")
    print(f"\n  Posterior predictive check:")
    print(f"    w̄:  média = {np.mean(w_post):.1f}, "
          f"mediana = {np.median(w_post):.1f} (obs: {W_OBS})")
    print(f"    L̄q: média = {np.mean(lq_post):.1f}, "
          f"mediana = {np.median(lq_post):.1f} (obs: {LQ_OBS})")
    print(f"\n  Tempo total Q1: {t1_q1 - t0_q1:.1f}s")


    # --- Gráficos Q1 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Posteriori marginal de λ
    ax = axes[0, 0]
    ax.hist(lam_post, bins=40, density=True, alpha=0.6,
            color='steelblue', edgecolor='white')
    kde_lam = gaussian_kde(lam_post)
    x_lam = np.linspace(lam_post.min() * 0.9, lam_post.max() * 1.1, 200)
    ax.plot(x_lam, kde_lam(x_lam), 'k-', lw=2, label='KDE')
    ax.axvline(np.median(lam_post), color='red', ls='--',
               label=f'Mediana = {np.median(lam_post):.4f}')
    ax.set_xlabel('λ (taxa de chegada, clientes/min)')
    ax.set_ylabel('Densidade')
    ax.set_title('Posteriori de λ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Posteriori marginal de μ
    ax = axes[0, 1]
    ax.hist(mu_post, bins=40, density=True, alpha=0.6,
            color='darkorange', edgecolor='white')
    kde_mu = gaussian_kde(mu_post)
    x_mu = np.linspace(mu_post.min() * 0.9, mu_post.max() * 1.1, 200)
    ax.plot(x_mu, kde_mu(x_mu), 'k-', lw=2, label='KDE')
    ax.axvline(np.median(mu_post), color='red', ls='--',
               label=f'Mediana = {np.median(mu_post):.4f}')
    ax.set_xlabel('μ (taxa de atendimento, atend./min)')
    ax.set_ylabel('Densidade')
    ax.set_title('Posteriori de μ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) Posteriori conjunta (scatter)
    ax = axes[1, 0]
    sc = ax.scatter(lam_post, mu_post, alpha=0.5, s=15,
                    c=dist_post, cmap='viridis_r', edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Distância')
    # Linha de estabilidade: λ = 2μ  ⟹  ρ = 1
    lam_plot = np.linspace(lam_post.min() * 0.8, lam_post.max() * 1.2, 100)
    ax.plot(lam_plot, lam_plot / 2, 'r--', alpha=0.6, label='ρ = 1 (λ = 2μ)')
    ax.set_xlabel('λ')
    ax.set_ylabel('μ')
    ax.set_title('Posteriori Conjunta (λ, μ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (4) Posterior Predictive Check
    ax = axes[1, 1]
    ax.scatter(w_post, lq_post, alpha=0.5, s=15, color='steelblue',
               edgecolors='none', label='Amostras aceitas')
    ax.scatter(W_OBS, LQ_OBS, s=150, c='red', marker='*', zorder=5,
               label=f'Observado ({W_OBS}, {LQ_OBS})')
    ax.set_xlabel('w̄ (tempo médio de espera, min)')
    ax.set_ylabel('L̄q (comprimento médio da fila)')
    ax.set_title('Posterior Predictive Check')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Questão 1 — ABC Rejection Sampling (3 estágios)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('q1_posterioris.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: q1_posterioris.png")
    plt.close()


    # =====================================================================
    # QUESTÃO 2 — TESTE DE HIPÓTESE INFORMAL
    # =====================================================================

    print("\n" + "=" * 60)
    print("QUESTÃO 2 — Teste Informal: H₀: 1/μ > 2/λ")
    print("=" * 60)

    # H₀: tempo médio de atendimento > dobro do tempo médio entre chegadas
    # H₀: 1/μ > 2/λ  ⟺  λ > 2μ  ⟺  ρ > 1 (sistema instável)

    tempo_atend = 1.0 / mu_post      # tempo médio de atendimento
    dobro_entre_cheg = 2.0 / lam_post  # dobro do tempo médio entre chegadas
    delta = tempo_atend - dobro_entre_cheg  # positivo ⟹ H₀ verdadeira

    p_h0 = np.mean(delta > 0)

    print(f"\n  H₀: 1/μ > 2·(1/λ)  ⟺  λ > 2μ  ⟺  ρ > 1 (fila instável)")
    print(f"\n  Definindo Δ = 1/μ − 2/λ (positivo ⟹ H₀)")
    print(f"    Média de Δ:   {np.mean(delta):.2f} min")
    print(f"    Mediana de Δ: {np.median(delta):.2f} min")
    print(f"    IC 95%: [{np.quantile(delta, 0.025):.2f}, "
          f"{np.quantile(delta, 0.975):.2f}]")
    print(f"\n  P(H₀ | dados) = P(Δ > 0) = {p_h0:.4f} ({p_h0*100:.2f}%)")

    if p_h0 < 0.05:
        veredito = "Rejeitamos H₀ informalmente (evidência forte contra)"
    elif p_h0 < 0.10:
        veredito = "Evidência moderada contra H₀"
    elif p_h0 > 0.90:
        veredito = "Não rejeitamos H₀ (evidência forte a favor)"
    else:
        veredito = "Evidência inconclusiva"
    print(f"  → {veredito}")

    # --- Gráfico Q2 ---
    plt.figure(figsize=(10, 6))
    plt.hist(delta, bins=40, density=True, alpha=0.6,
             color='teal', edgecolor='white')

    kde_d = gaussian_kde(delta)
    x_d = np.linspace(delta.min() - 2, delta.max() + 2, 300)
    y_d = kde_d(x_d)
    plt.plot(x_d, y_d, 'k-', lw=2, label='KDE')

    # Sombrear região onde H₀ é verdadeira (Δ > 0)
    mask_h0 = x_d >= 0
    if np.any(mask_h0):
        plt.fill_between(x_d[mask_h0], y_d[mask_h0], alpha=0.3, color='red',
                         label=f'Região H₀  (P = {p_h0:.4f})')

    plt.axvline(0, color='red', ls='--', lw=2, label='Δ = 0 (fronteira)')
    plt.xlabel('Δ = 1/μ − 2/λ  (minutos)')
    plt.ylabel('Densidade')
    plt.title('Questão 2 — Teste Informal de H₀: 1/μ > 2/λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q2_teste_hipotese.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: q2_teste_hipotese.png")
    plt.close()


    # =====================================================================
    # QUESTÃO 3 — NÚMERO DE CADEIRAS NECESSÁRIAS
    # =====================================================================

    print("\n" + "=" * 60)
    print("QUESTÃO 3 — Quantas cadeiras são necessárias?")
    print("=" * 60)

    t0_q3 = time.time()

    # Para cada amostra da posteriori, simula uma fila e calcula
    # o quantil 90% de Lq(t) ao longo do tempo.
    n_post = len(lam_post)
    cadeiras = np.zeros(n_post, dtype=int)

    print(f"  Simulando {n_post} filas (uma por amostra da posteriori)...")

    args_q3 = zip(lam_post, mu_post, [TEMPO_MAX]*n_post, [N_SERV]*n_post)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        cadeiras = list(executor.map(_worker_q3, args_q3, chunksize=max(1, n_post // 16)))
    cadeiras = np.array(cadeiras, dtype=int)

    t1_q3 = time.time()

    print(f"\n  Nº de cadeiras (quantil 90% de Lq(t) por simulação):")
    print(f"    Média   = {np.mean(cadeiras):.1f}")
    print(f"    Mediana = {np.median(cadeiras):.0f}")
    print(f"    IC 95%: [{np.quantile(cadeiras, 0.025):.0f}, "
          f"{np.quantile(cadeiras, 0.975):.0f}]")
    recomendacao = int(np.quantile(cadeiras, 0.95))
    print(f"    Recomendação conservadora (Q95): {recomendacao} cadeiras")
    print(f"  Tempo Q3: {t1_q3 - t0_q3:.1f}s")

    # --- Gráfico Q3 ---
    plt.figure(figsize=(10, 6))
    bins_cad = np.arange(max(0, cadeiras.min() - 0.5),
                         cadeiras.max() + 1.5, 1)
    plt.hist(cadeiras, bins=bins_cad, density=True, alpha=0.65,
             color='forestgreen', edgecolor='white')
    plt.axvline(np.median(cadeiras), color='red', ls='--', lw=2,
                label=f'Mediana = {np.median(cadeiras):.0f}')
    plt.axvline(recomendacao, color='darkred', ls=':', lw=2,
                label=f'Quantil 95% = {recomendacao}')
    plt.xlabel('Número de Cadeiras (C)')
    plt.ylabel('Frequência relativa')
    plt.title('Questão 3 — Cadeiras para que 90% do Tempo '
              'Todos Possam Sentar')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q3_cadeiras.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: q3_cadeiras.png")
    plt.close()


    # =====================================================================
    # QUESTÃO 4 — M/M/2 (FILA ÚNICA) vs. 2×M/M/1 (DUAS FILAS)
    # =====================================================================

    print("\n" + "=" * 60)
    print("QUESTÃO 4 — Fila Única (M/M/2) vs. Duas Filas (2×M/M/1)")
    print("=" * 60)

    t0_q4 = time.time()

    # Para cada amostra da posteriori, simulamos:
    #   a) M/M/2: uma fila, 2 servidores (cenário real)
    #   b) 2×M/M/1: duas filas independentes, cada uma com λ/2 e 1 servidor

    w_mm2_arr = np.zeros(n_post)
    w_2mm1_arr = np.zeros(n_post)

    print(f"  Simulando {n_post} pares de cenários...")

    args_q4 = zip(lam_post, mu_post, [2]*n_post, [1]*n_post)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_q4 = list(executor.map(_worker_q4, args_q4, chunksize=max(1, n_post // 16)))
    w_mm2_arr = np.array([r[0] for r in results_q4])
    w_2mm1_arr = np.array([r[1] for r in results_q4])

    # Diferença: positiva ⟹ fila única (M/M/2) é melhor
    delta_w = w_2mm1_arr - w_mm2_arr
    pct_mm2_melhor = np.mean(delta_w > 0) * 100

    t1_q4 = time.time()

    print(f"\n  Tempo médio de espera (min):")
    print(f"    M/M/2 (fila única):  "
          f"média = {np.mean(w_mm2_arr):.1f}, "
          f"mediana = {np.median(w_mm2_arr):.1f}")
    print(f"    2×M/M/1 (duas filas): "
          f"média = {np.mean(w_2mm1_arr):.1f}, "
          f"mediana = {np.median(w_2mm1_arr):.1f}")
    print(f"\n  Diferença Δw = w̄(2×M/M/1) − w̄(M/M/2):")
    print(f"    Média   = {np.mean(delta_w):.1f} min")
    print(f"    Mediana = {np.median(delta_w):.1f} min")
    print(f"    IC 95%: [{np.quantile(delta_w, 0.025):.1f}, "
          f"{np.quantile(delta_w, 0.975):.1f}]")
    print(f"\n  M/M/2 é melhor em {pct_mm2_melhor:.1f}% das simulações")
    print(f"  Tempo Q4: {t1_q4 - t0_q4:.1f}s")

    # --- Gráfico Q4 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(w_mm2_arr, bins=40, density=True, alpha=0.5,
            color='steelblue', label='M/M/2 (fila única)', edgecolor='white')
    ax.hist(w_2mm1_arr, bins=40, density=True, alpha=0.5,
            color='coral', label='2×M/M/1 (duas filas)', edgecolor='white')
    ax.set_xlabel('Tempo Médio de Espera (min)')
    ax.set_ylabel('Densidade')
    ax.set_title('Distribuição do Tempo de Espera')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(delta_w, bins=40, density=True, alpha=0.6,
            color='mediumpurple', edgecolor='white')
    ax.axvline(0, color='red', ls='--', lw=2, label='Sem diferença')
    ax.axvline(np.median(delta_w), color='black', ls='--', lw=1.5,
               label=f'Mediana = {np.median(delta_w):.1f} min')
    ax.set_xlabel('Δw = w̄(2×M/M/1) − w̄(M/M/2)  (min)')
    ax.set_ylabel('Densidade')
    ax.set_title('Diferença no Tempo de Espera')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Questão 4 — Fila Única vs. Duas Filas Independentes',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('q4_comparacao_filas.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: q4_comparacao_filas.png")
    plt.close()


    # =====================================================================
    # QUESTÃO 5 — DESISTÊNCIA DE CLIENTES (BALKING) — BÔNUS
    # =====================================================================

    print("\n" + "=" * 60)
    print("QUESTÃO 5 — Desistência de Clientes (Balking) — Bônus")
    print("=" * 60)

    t0_q5 = time.time()

    # Perfil escolhido justificado
    perfil = {'nome': 'Intermediário', 'alpha': 0.25, 'k0': 15}

    print(f"  Para estimar corretamente o número de desistências no dia\n"
          f"  observado, o próprio modelo generativo precisa incorporar\n"
          f"  a desistência, pois ela suprime Lq e W.")
    print(f"  Perfil assumido: {perfil['nome']} (α={perfil['alpha']}, k₀={perfil['k0']})")
    print(f"  Rodando ABC dedicado para o Balking (n=40.000)...\n")

    res_q5 = executar_abc_balking(
        n_amostras=40000,
        lam_range=(0.01, 3.5), # lam precisará ser maior pra atingir Lq=22 mesmo com desistência
        mu_range=(0.005, 1.0),
        frac_aceite=0.025, # reter os melhores 1000
        alpha=perfil['alpha'],
        k0=perfil['k0'],
        rotulo="ABC Balking"
    )

    lam_balk = res_q5['lam']
    mu_balk = res_q5['mu']
    desist_arr = res_q5['desistencias']
    potenc_arr = res_q5['potenciais']
    
    print(f"\n--- Resultados ABC com Balking ---")
    print(f"  ε final: {res_q5['eps']:.4f}")
    print(f"  λ real: {np.median(lam_balk):.4f} (muito maior do que sem balking)")
    print(f"  μ real: {np.median(mu_balk):.4f}")
    
    # Evitar divisão por zero
    taxas = np.where(potenc_arr > 0, desist_arr / potenc_arr, 0)

    print(f"\n  Estatísticas de Desistência:")
    print(f"    Desistências: média = {np.mean(desist_arr):.1f}, "
          f"mediana = {np.median(desist_arr):.0f}")
    print(f"    IC 95%: [{np.quantile(desist_arr, 0.025):.0f}, "
          f"{np.quantile(desist_arr, 0.975):.0f}]")
    print(f"    Taxa média de desistência: {100*np.mean(taxas):.1f}%")

    t1_q5 = time.time()
    print(f"\n  Tempo Q5: {t1_q5 - t0_q5:.1f}s")

    # --- Gráfico Q5: histograma único de desistências ---
    plt.figure(figsize=(7, 5))
    plt.hist(desist_arr, bins=30, density=True, alpha=0.7, color='#27ae60', edgecolor='white')
    plt.axvline(np.median(desist_arr), color='black', ls='--', lw=2, label=f'Mediana = {np.median(desist_arr):.0f}')
    plt.xlabel('Nº de Desistências no Dia')
    plt.ylabel('Densidade')
    plt.title(f"Questão 5 — Estimativa de Clientes que Desistiram\nPerfil: {perfil['nome']} (α={perfil['alpha']}, k₀={perfil['k0']})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q5_desistencias_real.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo: q5_desistencias_real.png")
    plt.close()

    # =====================================================================
    # RESUMO FINAL
    # =====================================================================

    t_total = time.time() - t0_q1

    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"  Tempo total: {t_total:.1f}s\n")
    print(f"  Q1: λ ≈ {np.median(lam_post):.4f} cl/min, "
          f"μ ≈ {np.median(mu_post):.4f} at/min, "
          f"ρ ≈ {np.median(rho_post):.4f}")
    print(f"  Q2: P(H₀: 1/μ > 2/λ | dados) = {p_h0:.4f}")
    print(f"  Q3: Cadeiras ≈ {np.median(cadeiras):.0f} (mediana), "
          f"{recomendacao} (conservador)")
    print(f"  Q4: M/M/2 melhor em {pct_mm2_melhor:.0f}% das vezes "
          f"(Δw mediana = {np.median(delta_w):.1f} min)")
    print(f"  Q5 (ABC p/ Balking): {np.median(desist_arr):.0f} desistências (mediana), "
          f"λ_real ≈ {np.median(lam_balk):.4f}")


if __name__ == '__main__':
    # Proteção p/ multiprocessamento no Windows
    main()
