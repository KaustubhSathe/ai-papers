# Get the directory where the script is located
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = "python"

# Yelp Reviews dataset
New-Item -ItemType Directory -Force -Path "$HERE\yelp" | Out-Null
if (-not (Test-Path "$HERE\yelp\raw_train.csv")) {
    & $PYTHON download.py "1xeUnqkhuzGGzZKThzPeXe2Vf6Uu_g_xM" "$HERE\yelp\raw_train.csv"
}
if (-not (Test-Path "$HERE\yelp\raw_test.csv")) {
    & $PYTHON download.py "1G42LXv72DrhK4QKJoFhabVL4IU6v2ZvB" "$HERE\yelp\raw_test.csv"
}
if (-not (Test-Path "$HERE\yelp\reviews_with_splits_lite.csv")) {
    & $PYTHON download.py "1Lmv4rsJiCWVs1nzs4ywA9YI-ADsTf6WB" "$HERE\yelp\reviews_with_splits_lite.csv"
}

# Surnames Dataset
New-Item -ItemType Directory -Force -Path "$HERE\surnames" | Out-Null
if (-not (Test-Path "$HERE\surnames\surnames.csv")) {
    & $PYTHON download.py "1MBiOU5UCaGpJw2keXAqOLL8PCJg_uZaU" "$HERE\surnames\surnames.csv"
}
if (-not (Test-Path "$HERE\surnames\surnames_with_splits.csv")) {
    & $PYTHON download.py "1T1la2tYO1O7XkMRawG8VcFcvtjbxDqU-" "$HERE\surnames\surnames_with_splits.csv"
}

# Books Dataset
New-Item -ItemType Directory -Force -Path "$HERE\books" | Out-Null
if (-not (Test-Path "$HERE\books\frankenstein.txt")) {
    & $PYTHON download.py "1XvNPAjooMyt6vdxknU9VO_ySAFR6LpAP" "$HERE\books\frankenstein.txt"
}
if (-not (Test-Path "$HERE\books\frankenstein_with_splits.csv")) {
    & $PYTHON download.py "1dRi4LQSFZHy40l7ZE85fSDqb3URqh1Om" "$HERE\books\frankenstein_with_splits.csv"
}

# AG News Dataset
New-Item -ItemType Directory -Force -Path "$HERE\ag_news" | Out-Null
if (-not (Test-Path "$HERE\ag_news\news.csv")) {
    & $PYTHON download.py "1hjAZJJVyez-tjaUSwQyMBMVbW68Kgyzn" "$HERE\ag_news\news.csv"
}
if (-not (Test-Path "$HERE\ag_news\news_with_splits.csv")) {
    & $PYTHON download.py "1Z4fOgvrNhcn6pYlOxrEuxrPNxT-bLh7T" "$HERE\ag_news\news_with_splits.csv"
}

# NMT Dataset
New-Item -ItemType Directory -Force -Path "$HERE\nmt" | Out-Null
if (-not (Test-Path "$HERE\nmt\eng-fra.txt")) {
    & $PYTHON download.py "1o2ac0EliUod63sYUdpow_Dh-OqS3hF5Z" "$HERE\nmt\eng-fra.txt"
}
if (-not (Test-Path "$HERE\nmt\simplest_eng_fra.csv")) {
    & $PYTHON download.py "1jLx6dZllBQ3LXZkCjZ4VciMQkZUInU10" "$HERE\nmt\simplest_eng_fra.csv"
} 