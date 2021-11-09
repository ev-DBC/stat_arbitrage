# Tools library

## Index

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

## Functions description 
*   <span id="rightdays">`rightdays (df, time_shift, datesIS, datesOS)`</span>

    This function sections the dataset into In Sample and Out of Sample taking as input the start and end dates of the two samples.
    **Inputs**:

         df: dataframe

         time_shift: numeric value for obtaining the right hours

         datesIS: training set dates

         datesOS: test set dates
    **Outputs**:

         [IS, OS]: IS and OS dataframe

*   <span id="righthoursIS">`righthoursIS (df, hours)`</span>

    This function selects the desired time slot of the trai set.
    **Inputs**:

         df: dataframe used 

         hours: time window to extract
    **Outputs**:

         dff: selected dataset

*   <span id="righthoursOS">`righthoursOS (df, hours)`</span>

    This function selects the desired time slot of the test set.
    **Inputs**:

         df: dataframe used

         hours: time window where both products are tradable
    **Outputs**:

         dff: dataframe selezionato
*   <span id="logratio">`logratio (df, cHO, cLGO)`</span>

    This function computes the log-ratio of the mid-prices between HO and LGO. Prices were rescaled to make the comparison consistent.
    **Inputs**:

         df: mid-prices dataframe

         cHO: conversion rate in barrels for HOc2

         cLGO: conversion rate in barrels for LGOc6
    **Outputs**:

         logratio: df with log-ratio column added in it

*   <span id="outliercheck">`outliercheck (df)`</span>

    This function receives a data vector as input and returns the vector cleaned of any outliers, the indices of the non-outlier data and the indices of the outliers in the original vector.

    A datapoint is considered a o:

    1.  It is less than (greater) than the first (third) quartile more than three times

        the interquantile range (IQR);
    2.  It is more than IQR away from the (i-1) -th datapoint and this distance 

        is recovered at least 95% from the (i + 1) -th datapoint.o.

    **Inputs**:

         df: dataframe
    **Outputs**:

         [df_OC, Outdf] = cleaned dataframe and removed outliers

*   <span id="MLE_estimator">`MLE_estimator (logratio, dt)`</span>

    This function takes the data and the time grid as input (assuming the data equally spaced in 24h) and returns the parameters k, eta and sigma for an OU process that most likely fitsthe dataset.
    **Inputs**:

         logratio: array cointaining log-ratios of the midprice

         dt: time grid 
    **Outputs**:

         [k, eta, sigma]: list containing the parameters of the OU model

*   <span id="costs">`costs (df)`</span>

    This function computes the transaction cost by assuming both the average cost of the IS sample.
    **Inputs**:

         df: IS dataframe
    **Outputs**:

         cost: transaction cost

*   <span id="handle">Functions section</span>

    Section consisting of functions (transcription in Python of function handles) used later in the calculation of the long-run return (mu) and levels (u and d) of the trading band.

*   <span id="long_run">`long_run (loss, cost, theta, SIGMA, leverage, c)`</span>

    This function returns the _u_ and _d_ levels that maximize the long-run return (_mu_), the _leverage_ used and the long-run return as 'Current function value' (printed in main) assuming that the log-prices follow the dynamics of a OU process of parameters _k_, (_eta_ = 0), _sigma_. Furthermore, the function returns the _leverage_ used in maximization: if _leverage_ = -1 is passed, the optimal value will be returned
    **Inputs**:

         loss: stop loss 

         cost: effective transaction costs
         theta: Transformation of a OU parameter (1/k)

         SIGMA: Transformation of a OU parameter (sigma/sqrt{2*k})

         leverage: leverage used, i.e. fraction of wealth invested in the risky asset (in decimals). If it is -1, the leverage considered is the optimal one.

         c: effective transaction cost expressed in _SIGMA_ units
    **Outputs**:

         [band, leverage]: list containing the trading band and _leverage_ used

*   <span id="generateOU">`generateOU (k, eta, sigma, x0, dt, N_step)`</span>

    This function simulates a sample data set (trajectory) of length _N_step_ that follows the dynamics of an OU process of known parameters (_k, eta, sigma_).
    **Inputs**:

         k: mean reversion speed

         eta: mean reverting value

         sigma: volatility

         x0: initial value

         dt: time grid

         N_step: number of datapoints in the trajectory
    **Outputs**:

         time_serie:  simulated trajectory

*   <span id="statisticalbootstrap">`statisticalbootstrap (k, eta, sigma, dt, N_sample, N_steps, x0, leverage, loss, cost, c)`</span>

    This function generates _N_sample_ via the `generateOU` function and computes for each _ [k, eta, sigma] _ via the` MLE_estimator` function. With them, the optimal trading bands are calculated for each value present in _leverage_. As output we have a list _parameters_ containing _N_sample_ of the form _ [k, eta, sigma] _ and a list containing a number of lists equal to the length of _leverage_ containing _N_sample_ optimal trading band.
    **Inputs**:

         k: mean reversion speed

         eta: mean reverting value

         sigma: volatility

         dt:  time grid

         N_sample:  number of trajectories to simulated

         N_steps_: number of datapoints in the trajectory

         x0: initial condition

         leverage: list containing leverages

         loss: stop loss 

         cost: effective transaction costs

         c:  effective transaction cost expressed in _SIGMA_ units
    **Outputs**:

         time_serie: simulated sample

*   <span id="tradingStrategy">`tradingStrategy (U, D, L, leverage, W0, time_strategy, OSS_OC, cost, eta)`</span>

    This function simulates the Trading Strategy, returning the log-return (normalized over time), the final wealth and the indices and values in which a position opens / closes (long or short).
    **Inputs**:

         U: level above which a long position with profit is closed (a 0-centered process is considered)

         D: level at which a long position is opened (a 0-centered process is considered)

         L: level below which a long position with loss is closed (it is considered a process centered in 0)

         leverage: leverage 

         W0: initial wealth


         time_strategy: time for which the strategy is implemented (fraction of a year in bus-days)

         OSS_OC: OS dataframe containing the logratio on which to test the strategy

         cost: effective transaction cost

         eta: mean reverting value
    **Outputs**:

         log_return: annualized log_return

         Wt: wealth at the end of the strategy

         check_in: list in which are reported respectively indices (with respect to vector X) and log-prices in which a position is opened.

         check_out_: list in which are reported respectively indices (with respect to vector X) and log-prices in which a position is closed.
