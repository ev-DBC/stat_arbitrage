# Tools library

## Indice

*   [`rightdays`](#rightdays)
*   [`righthoursIS`](#righthoursIS)
*   [`righthoursOS`](#righthoursOS)
*   [`logratio`](#logratio)
*   [`outliercheck`](#outliercheck)
*   [`MLE_estimator`](#MLE_estimator)
*   [`costs`](#costs)
*   [ Functions section](#handle)
*   [`long_run`](#long_run)
*   [`generateOU`](#generateOU)
*   [`statisticalbootstrap`](#statisticalbootstrap)
*   [`tradingStrategy`](#tradingStrategy)

## Descrizione funzioni 
*   <span id="rightdays">`rightdays (df, time_shift, datesIS, datesOS)`</span>

    Questa funzione seziona il dataset in In Sample e Out of Sample prendendo in input le date di inizio e di fine dei due campioni.
    **Inputs**:

         df: dataframe

         time_shift: valore numerico per ottenere le date corrette

         datesIS: date di inizio e fine di IS

         datesOS: date di inizio e fine di OS
    **Outputs**:

         [IS, OS]: IS and OS dataframe

*   <span id="righthoursIS">`righthoursIS (df, hours)`</span>

    Questa funzione seleziona la fascia oraria desiderata dell’IS.
    **Inputs**:

         df: dataframe da cui selezionare gli orari

         hours: lista con fascia oraria dove il mercato è più liquido
    **Outputs**:

         dff: dataframe selezionato

*   <span id="righthoursOS">`righthoursOS (df, hours)`</span>

    Questa funzione seleziona la fascia oraria desiderata dell’OS.
    **Inputs**:

         df: dataframe da cui selezionare gli orari

         hours: lista con fascia oraria dove sono entrambi tradabili
    **Outputs**:

         dff: dataframe selezionato
*   <span id="logratio">`logratio (df, cHO, cLGO)`</span>

    Questa funzione computa il log-rapporto dei mid-prices tra HO e LGO. I prezzi sono stati riscalati affinché il confronto fosse consistente.
    **Inputs**:

         df: dataframe contenente i mid-prices

         cHO: tasso di conversione in barili per HOc2

         cLGO: tasso di conversione in barili per LGOc6
    **Outputs**:

         logratio: df con l’aggiunta della colonna logratio

*   <span id="outliercheck">`outliercheck (df)`</span>

    Questa funzione riceve in input un vettore di dati e restituisce il vettore ripulito da eventuali outlier, gli indici dei dati non outlier e gli indici degli outlier nel vettore originale.

    Il dato i-esimo è considerato outlier in due casi:

    1.  È minore (maggiore) del primo (terzo) quartile per più di tre volte

        il range interquartile (IQR);
    2.  Dista più di IQR dal dato (i-1)-esimo e questa distanza viene

        recuperata almeno al 95% dal dato (i+1)-esimo.

    **Inputs**:

         df: dataframe da cui eliminare eventuali outliers
    **Outputs**:

         [df_OC, Outdf] = dataframe ripulito e outliers rimossi

*   <span id="MLE_estimator">`MLE_estimator (logratio, dt)`</span>

    Questa funzione prende in input i dati e la spaziatura temporale (assumendo i dati equispaziati nelle 24h) e restituisce i parametri k, eta e sigma per un processo OU che più verosimilmente ricalca il dataset.
    **Inputs**:

         logratio: colonna del dataframe contenente i log-rapporti dei mid-prices

         dt: spaziatura temporale
    **Outputs**:

         [k, eta, sigma]: lista contenente i parametri del modello

*   <span id="costs">`costs (df)`</span>

    Questa funzione computa il costo di transazione assumendo sia il costo medio del campione IS.
    **Inputs**:

         df: IS dataframe
    **Outputs**:

         cost: costo di transazione

*   <span id="handle">Functions section</span>

    Sezione costituita da funzioni (trascrizione in Python di function handles) utilizzate successivamente nel calcolo del long-run return (mu) e dei livelli (u e d) della trading band.

*   <span id="long_run">`long_run (loss, cost, theta, SIGMA, leverage, c)`</span>

    Questa funzione restituisce i livelli _u_ e _d_ che massimizzano il long-run return (_mu_), il _leverage_ utilizzato e il long-run return come ‘Current function value’ (printato nel main) assumendo che i log-prices seguano la dinamica di un processo OU di parametri _k_, (_eta_=0), _sigma_. Inoltre la funzione ritorna il _leverage_ utilizzato nella massimizzazione: se viene passato _leverage_ =  -1 verrà restituito il valore ottimale.
    **Inputs**:

         loss: stop loss considerata

         cost: costo di transazione effettivo (non espresso usando _SIGMA_ come unità)

        theta: parametro dipendente dai parametri di OU (1/k)

         SIGMA: parametro dipendente dai parametri di OU (sigma/sqrt{2*k})

         leverage: leverage utilizzato, i.e. frazione di ricchezza investita nell’asset rischioso (in decimali). Se è -1 il leverage considerato è quello ottimo.

         c: costo di transazione effettivo usando _SIGMA_ come unità)
    **Outputs**:

         [band, leverage]: lista contenente la trading band e _leverage_ utilizzato

*   <span id="generateOU">`generateOU (k, eta, sigma, x0, dt, N_step)`</span>

    Questa funzione simula un sample set di dati (traiettoria) di lunghezza _N_step_ che segue la dinamica di un processo OU di parametri noti (_k, eta, sigma_).
    **Inputs**:

         k: coefficiente di rilassamento

         eta: mean reverting value

         sigma: volatilità

         x0: condizione iniziale

         dt: spaziatura temporale tra due elementi consecutivi della traiettoria

         N_step: numero di passi per ogni traiettoria
    **Outputs**:

         time_serie:  traiettoria simulata

*   <span id="statisticalbootstrap">`statisticalbootstrap (k, eta, sigma, dt, N_sample, N_steps, x0, leverage, loss, cost, c)`</span>

    Questa funzione genera _N_sample_ tramite la funzione `generateOU` e computa per ognuno _[k, eta, sigma]_ tramite la funzione `MLE_estimator`. Con essi vengono computate, per ogni valore presente in _leverage_, le trading band ottimali. Come output si ha una lista _parameters_ contenente _N_sample_ della forma  _[k, eta, sigma]_ e una lista contenente un numero di liste pari alla lunghezza di _leverage_ contenenti _N_sample_ trading band ottimali.
    **Inputs**:

         k: coefficiente di rilassamento

         eta: mean reverting value

         sigma: volatilità

         dt:  spaziatura temporale tra due elementi consecutivi della traiettoria

         N_sample:  numero di sample da generare

         N_steps_: numero di passi per ogni traiettoria

         x0: condizione iniziale

         leverage: lista contenente i leverage

         loss: stop loss considerata

         cost: costo di transazione effettivo (non espresso usando _SIGMA_ come unità)

         c: costo di transazione effettivo usando _SIGMA_ come unità)
    **Outputs**:

         time_serie: sample simulato

*   <span id="tradingStrategy">`tradingStrategy (U, D, L, leverage, W0, time_strategy, OSS_OC, cost, eta)`</span>

    Questa funzione simula la Trading Strategy, restituendo il log-return (normalizzato sul tempo), la ricchezza finale e gli indici e i valori in cui si apre/chiude una posizione (lunga o corta che sia).
    **Inputs**:

         U: livello sopra il quale viene chiusa una posizione lunga con profitto (si considera un processo centrato in 0)

         D: livello a cui viene aperta una posizione lunga (si considera un processo centrato in 0)

         L: livello sotto il quale viene chiusa una posizione lunga con perdita (si considera un processo centrato in 0)

         leverage: leverage considerato

         W0: ricchezza iniziale


         time_strategy: tempo per cui viene attuata la strategia (frazione di anno in bus-days)

         OSS_OC: dataframe OS contenente i logratio su cui testare la strategia

         cost: costo per ogni transazione

         eta: mean reverting value
    **Outputs**:

         log_return: log-return normalizzato sul tempo

         Wt: ricchezza al termine della strategia

         check_in: lista in cui sono riportati rispettivamente indici (rispetto al vettore X) e log-prices in cui viene aperta una posizione.

         check_out_: lista in cui sono riportati rispettivamente indici (rispetto al vettore X) e log-prices in cui viene chiusa una posizione.
