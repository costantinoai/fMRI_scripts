def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate mean, confidence interval and SEM of a sample
    
    return: mean, CI lower, CI upper, SEM
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, se
