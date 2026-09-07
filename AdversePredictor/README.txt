This project predicts advserse price movements within the next 5 seconds. ie, whether or not a hypothetical passive buy execution is going to experience a price drop in the short
term event horizon.

The model utilizes XGBoost and has been trained on live NSE tick data containing Last Traded Price, Volume and Timestamp across various stocks like Maruti Suzuki, Axis Bank, ICICI Bank, etc. 

Data Information & Limitations:

    The data used contains live tick NSE stock data. The parameters it contains are:

    ltp(Last Traded Price)
    Volume(The amount traded)
    Timestamp

    Limitations:

        The data does not contain information around OBI, mainly bids, asks, etc.
        



The pipeline is as follows:

NSE Tick Data is read from PostgreSQL
      ↓
Timestamp normalization to the nearest second
      ↓
1-second aggregation of stocks having multiple entries at the same timestamp
      ↓
Engineering Features like volatility, momentum, etc. to appropriately train the model
      ↓
Adverse-event labeling
      ↓
Running XGBoost on the data
      ↓
Risk probability
      ↓
Risk ranking

Experiments run-> 
    Out-of-time-validation 
    Cross-stock-validation 
    Feature-ablation
    Baseline-comparison 

Performance Metrics:

    Out-of-Time Validation: Mean ROC-AUC of 0.797, with a
    18.45× lift in adverse events among the top 0.5% highest-risk predictions.

    Cross-Stock Validation: Mean ROC-AUC of 0.822, with a
    26.81× lift among the top 0.5% highest-risk predictions.

    Feature Ablation: Performance increased from 0.573 ROC-AUC
    using returns alone to 0.797 using the complete feature set.

    Baseline Comparison: XGBoost achieved 0.795 ROC-AUC, compared
    with 0.772 for Logistic Regression.




