### Zippare fa schifo
[Il link])(https://github.com/bazelbuild/bazel/issues/8981)

### Aggiornare le dipendenze
```
bazel run //:requirements.update
```

### esecuzione `py_binary`
```
bazel run //app
```

### creazione virtual environment
```
py -3.9 -m venv .venv
.venv\Scripts\activate
pip install -r requiremements_torch_lock.txt
```

# exploring-deblurring-CNN
Osservazioni sul modello di rete convoluzionale generativo basato sui Multi-Scale Module. Codice originale https://github.com/c-yn/ConvIR/tree/main

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