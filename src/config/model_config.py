# src/config/model_config.py

MODEL_CONFIG = {
    (18, 0.1, "nobias"): {
        "alphasc":2.5, "beta":0.1, "ninit":5, "iinit":4,
        "chbetaiinit":[2, 3, 3, 4], "imax0":1,
        "betas":[-0.1, 0.0, 0.1, 0.2],
        "calpha":80, "bound":1.0, "nseed":24, 
    },
    (18, 0.1, "bias_y"): {
        "alphasc":2.5, "beta":0.1, "ninit":5, "iinit":1,
        "chbetaiinit":[2, 3, 3, 4], "imax0":1,
        "betas":[-0.1, 0.0, 0.1, 0.2],
        "calpha":80, "bound":1.0, "nseed":24, 
    },
    (18, 0.5, "nobias"): {
        "alphasc":2.5, "beta":0.1, "ninit":5, "iinit":1,
        "chbetaiinit":[2, 3, 3, 4], "imax0":1,
        "betas":[-0.1, 0.0, 0.1, 0.2],
        "calpha":80, "bound":1.0, "nseed":24, 
    },
    (18, 0.5, "bias_y"): {
        "alphasc":2.5, "beta":0.1, "ninit":5, "iinit":1,
        "chbetaiinit":[2, 3, 3, 4], "imax0":1,
        "betas":[-0.1, 0.0, 0.1, 0.2],
        "calpha":80, "bound":1.0, "nseed":24, 
    },
    (18, 0.1, "bias_x"): {
        "alphasc":2.0, "beta":0.1, "ninit":5, "iinit":3,
        "chbetaiinit":[2, 3, 3, 4], "imax0":1,
        "betas":[-0.1, 0.0, 0.1, 0.2],
        "calpha":80, "bound":1.0, "nseed":24, 
    },

    (60, 0.1, "nobias"): {
        "alphasc":6.0, "beta":0.1, "ninit":5, "iinit":4, 
        "chbetaiinit":[1, 0, 3], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":39, 
    },
    (60, 0.1, "bias_x"): {
        "alphasc":1.0, "beta":0.1, "ninit":5, "iinit":3, 
        "chbetaiinit":[1, 0, 3], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":39, 
    },
    (60, 0.1, "bias_y"): {
        "alphasc":1.5, "beta":0.0, "ninit":5, "iinit":4, 
        "chbetaiinit":[1, 0, 3], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":39, 
    },
    (60, 0.5, "nobias"): {
        "alphasc":1.5, "beta":0.0, "ninit":5, "iinit":4, 
        "chbetaiinit":[1, 0, 3], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":39, 
    },
    (60, 0.5, "bias_y"): {
        "alphasc":1.5, "beta":0.0, "ninit":5, "iinit":4, 
        "chbetaiinit":[1, 0, 3], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":39, 
    },

    (210, 0.1, "nobias"): {
        "alphasc":1.5, "beta":0.0, "ninit":5, "iinit":0, 
        "chbetaiinit":[0, 4, 2], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":0, 
    },
    (210, 0.1, "bias_x"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":3, 
        "chbetaiinit":[0, 4, 2], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":0, 
    },
    (210, 0.1, "bias_y"): {
        "alphasc":0.5, "beta":-0.1, "ninit":5, "iinit":3, 
        "chbetaiinit":[0, 4, 2], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":0, 
    },
    (210, 0.5, "nobias"): {
        "alphasc":0.5, "beta":-0.1, "ninit":5, "iinit":3, 
        "chbetaiinit":[0, 4, 2], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":0, 
    },
    (210, 0.5, "bias_y"): {
        "alphasc":0.5, "beta":-0.1, "ninit":5, "iinit":3, 
        "chbetaiinit":[0, 4, 2], "imax0":2,
        "calpha":6, "bound":0.1, "nseed":0, 
    },

    (756, 0.1, "nobias"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":4,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3, 
    },
    (756, 0.2, "nobias"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":2,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.3, "nobias"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":0,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.4, "nobias"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":2,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.5, "nobias"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":2,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.1, "bias_y"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":4,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.2, "bias_y"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":4,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.3, "bias_y"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":4,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.4, "bias_y"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":4,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
#    (756, 0.5, "bias_y"): {
#        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":4,
#        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
#    },
    (756, 0.1, "bias_x"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":3, 
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.3, "bias_x"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":3,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },
    (756, 0.4, "bias_x"): {
        "alphasc":0.5, "beta":0.0, "ninit":5, "iinit":3,
        "calpha":2, "bound":0.1, "nseed":7, "imax0":3,
    },

#    (2772, 0.1, "nobias"): {
#        "alphasc":0.1, "beta":0.0, "ninit":5, "iinit":1,
#        "calpha":2, "bound":0.1, "nseed":9, "imax0":5,
#    },
#    (2772, 0.1, "bias_x"): {
#        "alphasc":0.1, "beta":0.0, "ninit":5, "iinit":1,
#        "calpha":2, "bound":0.1, "nseed":9, "imax0":5,
#    },
#    (2772, 0.1, "bias_y"): {
#        "alphasc":0.1, "beta":0.0, "ninit":5, "iinit":1,
#        "calpha":2, "bound":0.1, "nseed":9, "imax0":5,
#    },
#    (2772, 0.5, "bias_y"): {
#        "alphasc":0.1, "beta":0.0, "ninit":5, "iinit":1,
#        "calpha":2, "bound":0.1, "nseed":9,  "imax0":5,
#    },
#
#    (10296, 0.5, "bias_y"): {
#        "alphasc":0.1, "beta":0.0, "ninit":5, "iinit":1,
#        "calpha":2, "bound":0.1, "nseed":9, "imax0":5,
#    },
}
