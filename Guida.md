# Scaricare ed installare il progetto
Scaricare il repository al seguente [link](https://github.com/antonio-petrillo/Progetto_IoT_gruppo_20), oppure se si ha il comando `git` a disposizione sulla propria macchina:
```bash
git clone https://github.com/antonio-petrillo/Progetto_IoT_gruppo_20.git
```
Successivamente aprire un terminale nella progetto:
```bash
cd Progetto_IoT_Gruppo_20
```
Creare un [Virtual Environment (venv)](https://docs.python.org/3/library/venv.html) per python:
```bash
python -m venv venv
```

## Attivare il Virtual Environment
Il seguente comando (da scegliere in base al proprio sistema operativo) va eseguito ogni volta che si apre un nuovo terminale.

### Istruzioni Windows
```powershell
# indipendentemente dal tipo di shell utilizzata
.\venv\Scripts\activate 

# se il comando precedente non funziona e si utilizza il Command Prompt
.\venv\Scripts\activate.bat

# se il comando precedente non funziona e si utilizza i Powershell
.\venv\Scripts\activate.ps1
```

### Istruzioni Linux/MacOS/BSD
```bash
source ./venv/bin/activate
```

## Installare le dipendenze
Da eseguire solo una volta e dopo aver attivato il Virtual Environment (sempre nella root del progetto):
```powershell
python -m pip install -r requirements.txt
```



