/// This module contains the functions used for fitting the negative binomial
/// generalized linear model. Most of this module was derived directly from the
/// R/R-math source code written in C. The lack of documentation in the source
/// code can be attributed to the original source code. However, explanations of
/// the mathematical reasoning behind the code are provided in R's excellent
/// [documentation here](https://cran.r-project.org/web/packages/DPQ/vignettes/log1pmx-etc.pdf).
use statrs::function::gamma::ln_gamma;

const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;
const M_LN_2PI: f64 = 1.837877066409345483560659472811;

macro_rules! R_D_exp {
    ($log_p:ident, $x:expr) => {
        if $log_p {
            $x
        } else {
            $x.exp()
        }
    };
}

macro_rules! R_D__1 {
    ($log_p:ident) => {
        if $log_p {
            0.0
        } else {
            1.0
        }
    };
}

macro_rules! R_D__0 {
    ($log_p:ident) => {
        if $log_p {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    };
}

fn stirlerr(n: f64) -> f64 {
    const S0: f64 = 0.083333333333333333333; /* 1/12 */
    const S1: f64 = 0.00277777777777777777778; /* 1/360 */
    const S2: f64 = 0.00079365079365079365079365; /* 1/1260 */
    const S3: f64 = 0.000595238095238095238095238; /* 1/1680 */
    const S4: f64 = 0.0008417508417508417508417508; /* 1/1188 */

    /*
      exact values for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
    */
    const SFERR_HALVES: [f64; 31] = [
        0.0,                           /* n=0 - wrong, place holder only */
        0.1534264097200273452913848,   /* 0.5 */
        0.0810614667953272582196702,   /* 1.0 */
        0.0548141210519176538961390,   /* 1.5 */
        0.0413406959554092940938221,   /* 2.0 */
        0.03316287351993628748511048,  /* 2.5 */
        0.02767792568499833914878929,  /* 3.0 */
        0.02374616365629749597132920,  /* 3.5 */
        0.02079067210376509311152277,  /* 4.0 */
        0.01848845053267318523077934,  /* 4.5 */
        0.01664469118982119216319487,  /* 5.0 */
        0.01513497322191737887351255,  /* 5.5 */
        0.01387612882307074799874573,  /* 6.0 */
        0.01281046524292022692424986,  /* 6.5 */
        0.01189670994589177009505572,  /* 7.0 */
        0.01110455975820691732662991,  /* 7.5 */
        0.010411265261972096497478567, /* 8.0 */
        0.009799416126158803298389475, /* 8.5 */
        0.009255462182712732917728637, /* 9.0 */
        0.008768700134139385462952823, /* 9.5 */
        0.008330563433362871256469318, /* 10.0 */
        0.007934114564314020547248100, /* 10.5 */
        0.007573675487951840794972024, /* 11.0 */
        0.007244554301320383179543912, /* 11.5 */
        0.006942840107209529865664152, /* 12.0 */
        0.006665247032707682442354394, /* 12.5 */
        0.006408994188004207068439631, /* 13.0 */
        0.006171712263039457647532867, /* 13.5 */
        0.005951370112758847735624416, /* 14.0 */
        0.005746216513010115682023589, /* 14.5 */
        0.005554733551962801371038690, /* 15.0 */
    ];
    let nn;

    if n <= 15.0 {
        nn = n + n;
        if nn.fract() == 0.0 {
            return SFERR_HALVES[nn as usize];
        };
        return ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - M_LN_SQRT_2PI;
    }

    nn = n * n;
    return if n > 500.0 {
        (S0 - S1 / nn) / n
    } else if n > 80.0 {
        (S0 - (S1 - S2 / nn) / nn) / n
    } else if n > 35.0 {
        (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    } else {
        /* 15 < n <= 35 : */
        (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
    };
}

fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np); // might underflow to 0
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN {
            return s;
        };
        let mut ej = 2.0 * x * v;
        v *= v; // "v = v^2"
        for j in 1..1000 {
            /* Taylor series; 1000: no infinite loop
            as |v| < .1,  v^2000 is "zero" */
            ej *= v; // = 2 x v^(2j+1)
            let s_ = s;
            s += ej / ((j << 1) + 1) as f64;
            if s == s_ {
                eprintln!("bd0({x}, {np}): T.series w/ {j} terms -> bd0={s}");
                /* last term was effectively 0 */
                return s;
            }
        }
        eprintln!(
            "bd0({x}, {np}): T.series failed to converge in 1000 it.; s={s}, ej/(2j+1)={}",
            ej / ((1000 << 1) + 1) as f64,
        );
    }
    /* else:  | x - np |  is not too small */
    return x * (x / np).ln() + np - x;
}

fn dbinom_raw(x: f64, n: f64, p: f64, q: f64, log_p: bool) -> f64 {
    if p == 0.0 {
        return if x == 0.0 {
            R_D__1!(log_p)
        } else {
            R_D__0!(log_p)
        };
    };
    if q == 0.0 {
        return if x == n {
            R_D__1!(log_p)
        } else {
            R_D__0!(log_p)
        };
    };

    let lc;
    if x == 0.0 {
        if n == 0.0 {
            return R_D__1!(log_p);
        };
        lc = if p < 0.1 {
            -bd0(n, n * q) - n * p
        } else {
            n * q.ln()
        };
        return R_D_exp!(log_p, lc);
    }
    if x == n {
        lc = if q < 0.1 {
            -bd0(n, n * p) - n * q
        } else {
            n * p.ln()
        };
        return R_D_exp!(log_p, lc);
    }
    if x < 0.0 || x > n {
        return R_D__0!(log_p);
    };

    /* n*p or n*q can underflow to zero if n and p or q are small.  This
    used to occur in dbeta, and gives NaN as from R 2.3.0.  */
    lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);

    /* f = (M_2PI*x*(n-x))/n; could overflow or underflow */
    /* Upto R 2.7.1:
     * lf = log(M_2PI) + log(x) + log(n-x) - log(n);
     * -- following is much better for  x << n : */
    let lf = M_LN_2PI + x.ln() + (-x / n).ln_1p();

    return R_D_exp!(log_p, lc - 0.5 * lf);
}

pub fn rf_dnbinom_mu(x: f64, size: f64, mu: f64, log_p: bool) -> f64 {
    /* be accurate, both for n << mu, and n >> mu : */
    if x == 0.0 {
        let z = if size < mu {
            (size / (size + mu)).ln()
        } else {
            (-mu / (size + mu)).ln_1p()
        };
        return R_D_exp!(log_p, size * z);
    }
    if x < 1e-10 * size {
        /* don't use dbinom_raw() but MM's formula: */
        /* FIXME --- 1e-8 shows problem; rather use algdiv() from ./toms708.c */
        let prob = if size < mu {
            (size / (1.0 + (size / mu))).ln()
        } else {
            (mu / (1.0 + (mu / size))).ln()
        };
        return R_D_exp!(log_p, binom_sct(x, size, prob));
    } else {
        /* no unnecessary cancellation inside dbinom_raw, when
         * x_ = size and n_ = x+size are so close that n_ - x_ loses accuracy */
        let prob = size / (size + x);
        let ans = dbinom_raw(size, x + size, prob, 1.0 - prob, log_p);
        return if log_p { prob.ln() + ans } else { prob * ans };
    }
}

fn binom_sct(x: f64, size: f64, prob: f64) -> f64 {
    size * prob.ln() + x * (size.ln() + (-prob).ln_1p()) - ln_gamma(x + 1.0)
        + (x * (x - 1.0) / (2.0 * size)).ln_1p()
}
