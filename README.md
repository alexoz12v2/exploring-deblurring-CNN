# Debluring with Convolutional Neural Networks

Osservazioni sul modello di rete convoluzionale generativo basato sui Multi-Scale Module. Codice originale <https://github.com/c-yn/ConvIR/tree/main>

## Installare bazel

- Scaricare `choco`: Da powershell amministratore

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

- Scaricare `MSYS` e pacchetti usati da bazel: Andare su [Link installazione](https://msys2.org), scaricare ed esequire msys terminal, e installare i pacchetti

```sh
pacman -S zip unzip diffutils git patch
```

- settare variabile d'ambiente `BAZEL_SH` a `C:\path\a\msys64\usr\bin`
- Scaricare java
- Installare `bazelisk` con `choco`: Powershell amministratore

```powershell
choco install bazelisk
```

- Abilitare utente corrente alla creazione del symlinks: Win + R ed eseguire `gpedit.msc` (se hai windows home, allora segui [la guida](https://www.majorgeeks.com/content/page/enable_group_policy_editor_in_windows_10_home_edition.html))
   dopodiche recarsi in `ComputerConfiguration/Windows Settings/Security Settings/Local Policies/User Right Assignment` e alla policy `Create symbolic links` aggiungere
   l'utente corrente (puoi vedere una lista degli utenti con il comando da `cmd`: `net user`)
- Abilitare "Developer Mode" da pannello di controllo
- Abilitare Long Directory paths: Win + R, `regedit`, e andare sotto `HKEY_LOCAL_MACHINE/SYSTEM/CurrentControlSet/Control/FileSystem` e settare a DWORD 1 `LongPathsEnabled`
- *(Opzionale)* Scaricare il formatter `wingett install astral-sh.ruff`, il quale, permette di eseguire, dalla root della repository, il comando `ruff format` per
   formattare in automatico tutti i files `.py` secondo le regole specificate nel file `.ruff.toml`
- *(Opzionale)* Scaricare buildifier da [questo link](https://github.com/bazelbuild/buildtools/releases) (selezionare la versione windows amd64), rinominare il file in `buildifier.exe`,
   scaricarlo in un posto a piacere ed inserirlo nella variabile di ambiente `Path`.
   Questo fa si che l'estensione VSCode "bazel" possa individuare tutti i bazel targets

### Creare Kaggle Token

Andare su [Kaggle Account](https://www.kaggle.com/settings/account) e dopo aver fatto login, cliccare "Create New Token", il che apre un prompt per scaricare `kaggle.json`.
Piazzarlo in

- `$XDG_CONFIG_HOME/kaggle/kaggle.json` (Linux)
- `C:\Users\<Windows Username>\.kaggle\kaggle.json` (Windows, se hai spostato la `Users` directory usa quella)
- `~/.kaggle/kaggle.json` (Altro)
Una volta che `kaggle.json` sta al posto giusto, `kaggle.api.authenticate()` non dovrebbe dare problemi

### Zippare fa schifo

[Il link](https://github.com/bazelbuild/bazel/issues/8981)

### installare piu versioni

```powershell
winget install Python.Python.3.10
```

verificare con `py --list` che tutte le versioni installate sono visibili

### Aggiornare le dipendenze

```powershell
bazel run //:requirements.update
```

### esecuzione `py_binary`

```powershell
bazel run //app
```

### creazione virtual environment

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requiremements_torch_lock.txt
```

### Se bazel non aggiorna i files python

stai attento a non modificare i files nei runfiles, solo quelli nel workspace. Se bazel non vede le modifiche fatte su un file workspace,
rifai la build settndo una variabile d'ambiente a caso per forzare bazel a invalidare la build

```powershell
bazel build --action_env="avariable=1" :mytarget
```

In questo modo viene invalidato il convenience symlink `bazel-bin`, quindi cancella anche quello

### Struttura Repository

- `docs/`: Possibili file `.tex` contenenti testo sia per appunti ad uso personale che report per la consegna finale
- `edcnn/`: [python package](https://docs.python.org/3/tutorial/modules.html#packages) contenente la rete nelle sue varie declinazioni, assieme ai suoi pesi preallenati
- `scripts/`: vari file `.py` dotati di `if __name__ == '__main__'`, quindi scripts
- `tests/`: se le funzioni in `edcnn/` diventano particolarmente complicate, testarle con il package `unittest` ([Link](https://realpython.com/python-unittest/)) potrebbe essere utile
- `assets/`: cartella contenente, sotto altre sottocartelle, tutti i dataset utilizzati. All'interno di essi, ci si aspetta di
avere le sottocartelle `test/` e `train/`. Al momento, sono sono trackate da `git` per non inquinare il mio `git LFS Storage` :smile:

### Idee Possibili

- interfaccia spicciola per visualizzazione features con `dearpygui`, per il punto strumenti di visualizzazione
- Variazioni alla Loss Function (cambio dei pesi della somma pesata Dual-Domain, Cambio la trasformazione al dominio della frequenza, cambio lo spazio del colore)
- Prendere e applicare dei concetti dall'image processing. In particolare, analizzare il Blur?
- Introduzione di procedure di allenamento (eg combinando in qualche modo metodi non supervisionati?) la rete per dataset che non conosce (priority alta)

### Tasks

- [X] Proposal Accettato
- [ ] Far partire il codice del deblurring preallenato dalla repository del paper

:smile:

### Riferimenti Python Utili

- [NamedTuples](https://realpython.com/python-namedtuple/)
- Il `MultiShapeKernel` usa *Grouped Convolution*, spiegazione al [link](https://paperswithcode.com/method/grouped-convolution#:~:text=A%20Grouped%20Convolution%20uses%20a,level%20and%20high%20level%20features.)

### Dalla command line

Windows

```powershell
set BAZEL_FIX_DIR="fds" && set BAZEL_PYWIN_REMAP="fsd" && .\bazel-bin\app\app.exe --window
```
