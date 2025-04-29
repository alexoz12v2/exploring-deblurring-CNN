"""Layers rimaneggiati del paper di ConvIR
(riferisciti a https://animatedai.github.io/ per un'immagine mentale della convoluzione)
Rilevante e' l'utilizzo delle funzioni dal modulo torch.nn.functional, in particolare
- torch.nn.functional.unfold = immagina l'operazione di convoluzione, come il kernel avanza spazialmente lungo
  il tensore in input. Tale operazione e' equivalente ad una moltiplicazione matrice vettore.
  Dati Matrice (n x n)  [ a11 a12 a13 ] e kernel (k x k)
                    A = [ a21 a22 a23 ] K = [ k11 k12 ]
                        [ a31 a32 a33 ]     [ k21 k22 ]
  Per ogni riga del kernel, sovrapponi la riga in ogni posizione spaziale lungo una riga (a caso) del tensore in input
  e segnati una riga lunga n pari a * l'elemento del kernel se c'e', * altrimenti zero. Ripeti questa procedura per
  ogni riga del kernel
                 K1 = [ k11 k12 0   ]  K2 = [ k21 k22 0   ]
                      [ 0   k11 k12 ]       [ 0   k21 k22 ]
                  ottieni una colonna per ogni elemento lungo una riga di A (tensore in input) => n
                  ottieni una colonna per ogni possibile posizione spaziale del kernel lungo una riga => t = n-k+1
  Concatena lungo la riga tutti i contributi di ogni riga del kernel
                 K' = [ k11 k12 0   k21 k22 0   ]
                      [ 0   k11 k12 0   k21 k22 ]
  adesso fai stacking lungo la verticale della matrice ottenuta e aggiungi padding tale da arrivare ad una matrice
  t^2 x n^2 (in particolare aggiungi la matrice K' sotto se stessa, displaced lungo la riga di n posti)
                  M = [ k11 k12 0   k21 k22 0   -   -   -   ]
                      [ 0   k11 k12 0   k21 k22 -   -   -   ]  ho segnato a "-" gli zeri aggiunti come padding
                      [ -   -   -   k11 k12 0   k21 k22 0   ]
                      [ -   -   -   0   k11 k12 0   k21 k22 ]
  dopodiche ottieni un vettore n^2 concatenando tutte le righe del tensore di partenza
                  v = [ a11 a12 a13 a21 a22 a23 a31 a32 a33 ]^T
  La convoluzione (A * K) e' equivalente al prodotto riga-colonna (M v) [ https://www.baeldung.com/cs/convolution-matrix-multiplication ]
  Tornando alla posizione di unfolding (anche detta im2col o pixel unshuffle): dato un tensore 4D, in cui nella prima dimensione memorizzi gli esempi in
  un minibatch, fai la seguente operazione: Per ogni esempio (tensore 3D) nel minibatch:
  - posiziona in kernel sopra il tensore di input. Piuttosto che fare il prodotto scalare nella regione di attivazione
    prendi tutti gli elementi che il kernel copre e concatenali lungo l'asse dei canali. Dunque se hai un kernel 3x3
    e una immagine a 3 canali, il kernel vede 27 elementi, e li metti nell'asse dei canali del risultato
  - Continua cosi' per ogni posizione spaziale del kernel (t = n - k + 1). Ci sono t^2 posizioni spaziali
    (tensore in input n x n e kernel k x k, senza considerare padding, diluition o stride), allungandoti lungo una sola
    dimensione spaziale
    Esempio: unfold((10, 3, 244, 244), ker = 3x3) = (10, 3 * (3x3), (244 - 3 + 1)^2) = (10, 27, 58564)
    https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html estende la formula per stride, padding e diluition
  Perche' e' utile? Considera anche la sua operazione inversa col2im (o fold o pixel shuffle), la quale prende in input un tensore sputato da
  unfold (im2col), ovvero (num_batches x patch_column x spatial_positions) e distribuisce ciascuna patch column in un
  blocco (kernel_size x kernel_size x num_channel). Questo vuol dire che deve essere soddisfatta la condizione
      patch_column | (kernel_size)^2                                                       (| "e' divisibile". Il discorso vale anche per kernel rettangolari)
  perche' hai bisogno di riempire con lo stesso numero di canali tutti i posti fornite dal kernel per ogni locazione spaziale

  una seconda condizione e' che, dato l'output size desiderato (out_width x out_height), e la kernel_size,
  calcolati le posizioni spaziali che derivano dalla output size desiderata, ovvero
              (out_width - kernel_size + 1) x (out_height - kernel_size + 1) = required_spatial_positions        (la formula puo' essere espansa per includere diluition, stride, padding)
  la seconda condizione e' dunque            -> required_spatial_positions == spatial_positions
  Il fold, quando riassembla l'immagine, non lo fa in modo uguale. In particolare, *somma le posizioni overlapping*, effettivamente compiendo la convoluzione per un kernel di kernel_size x kernel_size con tutti gli elementi a uno
              fold(unfold(input)) = torch.ones() * input
  dunque unfold/fold dell'input con la stessa kernel size ti da la convoluzione
  - Questo significa che usare un kernel (1x1) nella operazione di fold ti riporta nella rappresentazione normale del tensore

  Interessante e' guardare al Pooling nella rappresentazione unfolded del tensore:
  - unfold e fold cercando di diminuire la size, rispettando le due condizioni

  La rappresentazione Unfolded del tensore torna utile se un layer deve applicare tanti filtri, perche nella forma unfolded
  convoluzione = moltiplicazione (vedi la spiegazione di sopra)
  Per esempio, se voglio applicare un filtro passa basso ad una immagine,
  - ne faccio la rappresentazione unfolded con kernel_size pari alla size del filtro passa basso (eg. Gaussian Blur)
  - linearizza lungo la riga il kernel. Adesso hai unfolded: (patch_column x spatial_positions) e flat_kernel: (spatial_positions x 1) <- (oppure spatial_positions x num_kernels)
  - applichi la convoluzione con una moltiplicazione matrice vettore!
  - fold con kernel_size 1x1 per tornare nella rappresentazione normale del vettore
  link di riferimento: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
"""

import enum
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import logging
from typing import NamedTuple, Tuple


class BasicConv(nn.Module):
    """BasicConv: Classe implementante un layer convolutivo (sia in downsampling che in transpose) sandwitched da operazioni comuni, ovvero
    - Batch Normalization
    - Nonlinearity di tipo ReLU
    Attributes:
        main: nn.Sequential contenente le operazioni compiute dal layer convolutivo
    """

    class Input(NamedTuple):
        """BasicConv.Input: Parametri del costruttore di BasicConv.

        Attributes:
            in_channel: Numero canali in input al layer convolutivo.
            out_channel: Numero canali in output al layer convolutivo.
            kernel_size: Dimensioni spaziali del kernel del layer convolutivo.
            stride: Displacement in orizzontale e verticale del kernel nell'input da una locazione spaziale alla successiva.
            bias: Booleano indicante se aggiungere il termine di bias in ogni convoluzione o no.
            norm: Booleano indicante se fare la batch normalization o meno.
            relu: Booleano indicante se applicare nonLinearity o meno.
            transpose: Booleano indicante se applicare convoluzione in downsampling o convoluzione trasposta.
            device: Device in cui i nn.Parameter del nn.Module risiedono.
        """

        in_channel: int
        """BasicConv.Input.in_channel: Numero canali in input al layer convolutivo."""

        out_channel: int
        """BasicConv.Input.out_channel: Numero canali in output al layer convolutivo."""

        kernel_size: int
        """BasicConv.Input.kernel_size: Dimensioni spaziali del kernel del layer convolutivo."""

        stride: int | Tuple[int, int]
        """BasicConv.Input.stride: Displacement in orizzontale e verticale del kernel nell'input da una locazione spaziale alla successiva."""

        bias: bool = True
        """BasicConv.Input.bias: Booleano indicante se aggiungere il termine di bias in ogni convoluzione o no."""

        norm: bool = False
        """BasicConv.Input.norm: Booleano indicante se fare la batch normalization o meno."""

        relu: bool = True
        """BasicConv.Input.relu: Booleano indicante se applicare nonLinearity o meno."""

        transpose: bool = False
        """BasicConv.Input.transpose: Booleano indicante se applicare convoluzione in downsampling o convoluzione trasposta."""

        device: torch.device = torch.device("cpu")
        """BasicConv.Input.device: Device in cui i nn.Parameter del nn.Module risiedono."""

    def __init__(self, conf: Input) -> None:
        """BasicConv.__init__: Costruzione del torch.nn.Sequential implementante il layer convolutivo
        Args:
            conf (BasicConv.Input): Configurazione del layer convolutivo.
        Returns:
            None
        """

        super(BasicConv, self).__init__()
        bias = True
        if conf.bias and conf.norm:
            bias = False

        layers = list()
        if conf.transpose:
            # aggiungi padding preciso per preservare la dimensione spaziale con stride unitario
            padding = conf.kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    conf.in_channel,
                    conf.out_channel,
                    conf.kernel_size,
                    padding=padding,
                    stride=conf.stride,
                    bias=bias,
                )
            )
        else:
            # aggiungi padding preciso per preservare la dimensione spaziale con stride unitario
            padding = conf.kernel_size // 2
            layers.append(
                nn.Conv2d(
                    conf.in_channel,
                    conf.out_channel,
                    conf.kernel_size,
                    padding=padding,
                    stride=conf.stride,
                    bias=bias,
                )
            )
        if conf.norm:
            layers.append(nn.BatchNorm2d(conf.out_channel))
        if conf.relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward: implementa il forward pass per Basic Conv, prende un minibatch x e ne calcola l'uscita y
        Args:
            x (torch.Tensor): Minibatch in input
        Returns:
            (torch.Tensor) output del layer dato il minibatch
        """

        return self.main.forward(x)


class DilatedAttention(nn.Module):
    """DilatedAttention: nn.Module dedicato all'applicazione di DSA (Dilated Square Attention)
    o DRA (Dilated Rectangle Attention). nome Originale: dynamic_filter o spatial_strip_att.
    Attributes:
        kernel_size (int): dimensione del kernel di convoluzione per il filtraggio, utilizzato nella fase im2col
        group (int): numero gruppi applicati durante la convoluzione per filtro passa basso
        dilation (int): diluition alla applicazione del filtro passa basso, fatto passando diluition a im2col
        conv (torch.nn.Conv2d): strato convolutivo 1x1 fatto per fare bottleneck a (group x kernel_size^2) canali, senza biases
        bn (torch.nn.BatchNorm2d): strato di batch normalization spaziale con numero features a (group x kernel_size^2)
        act (torch.nn.Tanh): funzione di attivazione tipica delle reti generative
        lamb_l (nn.Parameter): peso, channel wise, componente basse frequenze
        lamb_h (nn.Parameter): peso, channel wise, componente alte frequenze
        pad (torch.nn.ReflectionPad2d): padding per preservare size, applicato con strategia di riflesso del contenuto del bordo
        ap (torch.nn.AdaptiveAvgPool2d): adaptive average pooling fare media in tutte le posizioni spaziali, dando un colore/feature in output.
          Adaptive perche al posto di specificare la kernel size specifichi la output size. applicato all'inizio per stimare il filtro passa basso
        gap (nn.AdaptiveAvgPool2d): Adaptive average pooling applicato al residuale nel calcolo della stima del contenuto
          a bassa frequenza. Tali medie sono calcolate lungo tutte le posizioni spazionali nel DSA, o lungo la sola
          direzione orizzontale o verticale nel DRA
        inside_all (nn.Parameter): Il risultato del filtro passa basso viene rimodulato `low * (inside_all + 1) - inside_all * gap(input)`
          al fine di far imparare alla rete lo stacco delle basse frequenze dalla media dell'immagine di input (calcolata con il GAP). Cio' e'
          mirato al calcolo delle basse frequenze del volume di input senza dover passare al dominio delle frequenze
    """

    class SpatialMode(enum.Enum):
        """DilatedAttention.SpatialMode: Enum rappresentante le direzioni spaziali lungo cui si applica il filtro"""

        DILATED_SQUARE_ATTENTION = 1
        """DilatedAttention.SpatialMode.DILATED_SQUARE_ATTENTION: Applicazione filtro in un patch quadrato"""

        DILATED_RECTAGLE_ATTENTION_WIDTH = 2
        """DilatedAttention.SpatialMode.DILATED_RECTAGLE_ATTENTION_WIDTH: Applicazione del filtro in un patch rettangolare (diluition * kernel_size x 1)"""

        DILATED_RECTAGLE_ATTENTION_HEIGHT = 3
        """DilatedAttention.SpatialMode.DILATED_RECTAGLE_ATTENTION_HEIGHT: Applicazione del filtro in un patch rettangolare (1 x diluition * kernel_size)"""

    class Input(NamedTuple):
        """DilatedAttention.Input: Parametri del costruttore di DilatedAttention
        Attributes:
            inchannels (int): Numero di canali del volume di entrata atteso
            mode (DilatedAttention.SpatialMode): Modalita spaziale di applicazione dei filtro passa basso e passa alto convolutivi
            kernel_size (int): Size del kernel di convoluzione su cui si basano i filtri applicati, lungo le direzioni abilitate da `mode` (default = 3)
            dilation (int): Diluition del kernel di convoluzione su cui si basano i filtri applicati (default = 1)
            group (int): Numero Gruppi del kernel di convoulzione su cui si basano i filtri applicati (default = 8)
            device (torch.device): Device in cui i parametri del layer risiedono
        """

        inchannels: int
        """DilatedAttention.Input.inchannels (int): Numero di canali del volume di entrata atteso"""

        mode: "DilatedAttention.SpatialMode"
        """DilatedAttention.Input.mode (DilatedAttention.SpatialMode): Modalita spaziale di applicazione dei filtro passa basso e passa alto convolutivi"""

        kernel_size: int = 3
        """DilatedAttention.Input.kernel_size (int): Size del kernel di convoluzione su cui si basano i filtri applicati, lungo le direzioni abilitate da `mode` (default = 3)"""

        dilation: int = 1
        """DilatedAttention.Input.dilation (int): Diluition del kernel di convoluzione su cui si basano i filtri applicati (default = 1)"""

        group: int = 8
        """DilatedAttention.Input.group (int): Numero Gruppi del kernel di convoulzione su cui si basano i filtri applicati (default = 8)"""

        device: torch.device = torch.device("cpu")
        """DilatedAttention.Input.device (torch.device): Device in cui i parametri del layer risiedono"""

    _mode_dict = {
        SpatialMode.DILATED_SQUARE_ATTENTION: (1, 1),  # media su entrambi assi
        SpatialMode.DILATED_RECTAGLE_ATTENTION_WIDTH: (1, None),  # media in width
        SpatialMode.DILATED_RECTAGLE_ATTENTION_HEIGHT: (None, 1),  # media in height
    }
    """DilatedAttention._mode_dict: dict contenente tutte le associazioni tra le possibili modalita' di applicazioni dei filtri e 
    gli argomenti da passare alla funzione torch.nn.AdaptiveAvgPool2d
    """

    def __init__(self, conf: Input):
        """DilatedAttention.__init__: Costruzione dei componenti necessari ad applicare DRA/DSA. Nota che nel caso di
        DSA, manca la convoluzione 3x3 iniziale, questo perche' essa e' previamente applicata nella classe
        Args:
            conf (BasicConv.Input): Configurazione del layer .
        Returns:
            None
        """

        super(DilatedAttention, self).__init__()
        self.kernel_size = conf.kernel_size
        self.group = conf.group
        self.dilation = conf.dilation
        # bottleneck con no bias
        self.conv = nn.Conv2d(
            conf.inchannels,
            conf.group * conf.kernel_size**2,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(conf.group * conf.kernel_size**2)
        self.act = nn.Tanh()

        # inizializza i parametri del self.conv della rete secondo il criterio "He Initialization"
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        self.lamb_l = nn.Parameter(
            torch.zeros(conf.inchannels), requires_grad=True
        )  # peso low pass filter
        self.lamb_h = nn.Parameter(
            torch.zeros(conf.inchannels), requires_grad=True
        )  # peso high pass filter
        self.pad = nn.ReflectionPad2d(
            self.dilation * (conf.kernel_size - 1) // 2
        )  # calcola padding per preservare size

        # x = torch.randn(10, 3, 4, 4)                torch.Size([10, 3, 4, 4])
        # gap_all = AdaptiveAvgPool2d(1)(x)           torch.Size([10, 3, 1, 1])
        # gap_w = AdaptiveAvgPool2d((1, None))(x)     torch.Size([10, 3, 1, 4])
        # gap_h = AdaptiveAvgPool2d((None, 1))(x)     torch.Size([10, 3, 4, 1])
        self.ap = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # sti due sono uguali ma con parametri diversi
        self.gap = nn.AdaptiveAvgPool2d(DilatedAttention._mode_dict.get(conf.mode, 1))

        # Parametro che modula la componente continua estratta con un peso per ogni canale, disinteressandosi della
        # posizione spaziale, da cui il suffisso "all"
        self.inside_all = nn.Parameter(
            torch.zeros(conf.inchannels, 1, 1), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DilatedAttention.forward: implementa il forward pass per DilatedAttention, prende un minibatch x e ne calcola l'uscita
        Args:
            x (torch.Tensor): Minibatch in input, ndims = 4, con x.shape[1] == conf.inchannels
        Returns:
            (torch.Tensor) output del layer dato il minibatch
        """

        identity_input = x

        # 1: Low pass frequency filter extraction
        low_filter = self.ap(x)  # (c = 3) prendi il colore medio per ogni x_i in minibatch
        low_filter = self.conv(low_filter)  # convoluzione 1x1 per avere 'group x kernel_size^2' canali
        low_filter = self.bn(low_filter)  # dopo ogni convoluzione BatchNorm ci sta (DRA originale non la fa)
        # low filter adesso contiene la media spaziale del volume, bottlenecked al numero di canali che mi serve
        # n x (group x kernel_size^2) x 1 x 1

        # 2: im2col dell'input, seguito dalla scomposizione dei vari canali in gruppi. quindi passo a tensore 5D
        #    riferimento per grouped convolutions: https://www.youtube.com/watch?v=vVaRhZXovbw&t=12s
        #    essenzialmente, fare grouped convolution e' equivalente a splittare il input volume in num_groups volumi,
        #    fare convoluzione con ciascuno di essi e concatenare i risultati. Serve ad abbassare il numero di parametri
        #    e l'ammontare di computazione. Il motivo per cui passiamo al dominio delle patches e' spiegato all'inizio
        #    del file, ovvero voglio fare convoluzione manuale come moltiplicazione matriciale
        #    Dunque questa formula di reshaping e' la traduzione nel dominio dei tensori unfolded della grouped convolution
        # le spatial positions sono esattamente width x height per via del padding applicato
        # noto come al posto di passare ad un tensore 3D ne passo ad uno 5D, perche' mantengo la separazione dei canali
        n, x_c, x_h, x_w = x.shape
        x = nnfunc.unfold(
            self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation
        ).reshape(  # -> xn c*k2 hw
            n, self.group, x_c // self.group, self.kernel_size**2, x_h * x_w
        )  # -> xn g c/g k2 hw
        # Questa operazione ha la duplice funzione di facilitare 'Grouped Operations' e 'Operazioni Patch-Wise' (implementando il concetto di spatial attention)
        # Grouped Operations   = Suddividere i canali in gruppi e processare ciascun gruppo in parallelo
        # Patch-Wise Filtering = Allinea le colonne con la distribuzione spaziale del filtro

        # 3: Prepara anche il low pass filter per fare grouped convolutions
        l_n, low_filter_c, low_filter_h, low_filter_w = (
            low_filter.shape
        )  # dalla size filter scritta sopra passo a ...
        if l_n != n or low_filter_h * low_filter_w != 1:
            raise ValueError("io della vita non ho capito un")

        # ricorda che unsqueeze aggiunge una dimensione pari a 1 nell'indice che gli dai. Esempio con x di size 10 3 224 224
        # { 'ini': x.shape, '0': x.unsqueeze(0).shape, '1': x.unsqueeze(1).shape, '2': x.unsqueeze(2).shape, '3': x.unsqueeze(3).shape }
        # { 'ini': [10, 3, 224, 224], '0': [1, 10, 3, 224, 224], '1': [10, 1, 3, 224, 224], '2': [10, 3, 1, 224, 224]), '3': [10, 3, 224, 1, 224] }
        low_filter = low_filter.reshape(  # ... n, ((group x kernel_size^2) / kernel_size^2), kernel_size^2, (1 x 1)
            n,
            low_filter_c // self.kernel_size**2,
            self.kernel_size**2,
            low_filter_h * low_filter_w,
        ).unsqueeze(2)  # ... n, group, 1, kernel_size^2, 1
        # hai ottenuto una struttura analoga all'input (n, group, channels/group, kernel_size^2, height x width)
        low_filter = self.act(low_filter)  # passa filtro su nonlinearita tanh, caratterizzata da smoothness, bounds, simmetria -> regolarizzazione e controllo

        # 4: Applicazione filtro passa basso (10 img 224 x 224, 3 canali in 3 gruppi)
        # x = torch.randn(10, 3, 1, 9, 224 * 224)
        # k = torch.randn(10, 3, 1, 9, 1)
        # prod = x * k # nota che puoi fare il prodotto finche k e' broadcastabile (ciascuna dimensione in x multipla della corrispettiva in k)
        # x.shape, k.shape, prod.shape                      (224 * 224 = 50176)
        # >> (torch.Size([10, 3, 1, 9, 50176]), torch.Size([10, 3, 1, 9, 1]), torch.Size([10, 3, 1, 9, 50176]))
        # il broadcast in questo caso e' fatto su tutti i canali e posizioni spaziali, applicando il filtro passa basso in one shot
        # dopodiche fa la somma su tutti i prodotti per il kernel, completando il prodotto scalare, e poi riassembla tensore con reshape
        # dot_prod = torch.sum(prod, dim=3)
        # dot_prod.shape
        # >> torch.Size([10, 3, 1, 50176])
        # dopo il reshape
        # >> torch.Size([10, 3, 224, 224])
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, x_c, x_h, x_w)

        # 5: Estrazione componente a bassa frequenza
        # Dato il risultato del filtro passa basso, normalizzane il contenuto sottraendo alla versione amplificata  (di 1 + k_imparabile)
        # delle basse frequenze una componente continua dell'immagine calcolata come k_imparabile * colore_medio
        # per normalizzane si intende calcola le componenti a bassa frequenza dell'segnale come distanza pesata dal suo
        # valore DC, calcolato con GAP, pesata sul parametro imparabile inside_all
        # x = torch.randn(10, 3, 224, 224)
        # m = torch.randn(10, 3, 1, 1)
        # inside_all = torch.randn(3, 1, 1)
        # summand0 = x * (inside_all + 1.)
        # summand1 = - inside_all * m
        # summand0.shape, summand1.shape
        # >> (torch.Size([10, 3, 224, 224]), torch.Size([10, 3, 1, 1]))
        # Durante la fase di training la rete essenzialmente impara a bilanciare i suoi parametri tale che out_low
        # rappresenti la componente a bassa frequenza dell'immagine, come se fosse stata applicata la trasformata di Fourier
        out_low = low_part * (
            self.inside_all + 1.0
        ) - self.inside_all * self.gap(identity_input)

        # 6: Scala delle basse frequenze con un parametro imparabile. Dato che il lamb_l e' un tensore con size (C),
        #    aggiungiamo delle singleton dimensions mettendo dei None nello schema di indexing, e trasferiamo le C
        #    nella dimensione 1 per matchare quello che arriva da out_low
        out_low = out_low * self.lamb_l[None, :, None, None]

        # 7: Stima delle alte frequenze == Amplificazione del segnale con un parametro imparabile
        #    e riscalale di un fattore lamb_h + 1. (+ 1 perche il parametro parte a zero, e da li la rete impara un displacement dall'unita)
        #    Invece di calcolare le frequenze alte come residuo tra immagine e frequenze basse, che puo' introdurre problemi per via della sottrazione
        #    sto semplificando e calcolando frequenze alte come immagine amplificata
        #    Stiamo assumendo che le frequenze alte siano preservate nel segnale di input grezzo
        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.0)

        # 8: somma Frequenze basse estratte e amplificate a frequenze alte
        return out_low + out_high


class DRAWidthHeight(nn.Module):
    """DRAWidthHeight: Classe che implementa l'applicazione sequenziale del DRA lungo l'orizzontale e del DRA lungo la verticale
    originale: `spatial_strip_att`
    Attributes:
        H_spatial_att (DilatedAttention): layer DRA lungo verticale
        W_spatial_att (DilatedAttention): layer DRA lungo orizzontale
        gamma (torch.nn.Parameter): peso dell'output dalla sequenza dei due DRA
        beta (torch.nn.Parameter): peso dell'input sommato all'output dalla sequenza dei due DRA (blocco residuale)
    """

    class Input(NamedTuple):
        """DRAWidthHeight.Input: Parametri del costruttore di DRAWidthHeight
        Attributes:
            dim (int): numero di canali in input attesi
            group (int): numero di gruppi su cui applicare la convoluzione per il filtro passa basso
            dilation (int): dilation per il kernel del filtro passa basso
            kernel (int): dimensione del kernel per la convoluzione del filtro passa basso
            device (torch.device): Device in cui i parametri della rete risiedono
        """

        dim: int
        """DRAWidthHeight.Input.dim (int): numero di canali in input attesi"""

        group: int
        """DRAWidthHeight.Input.group (int): numero di gruppi su cui applicare la convoluzione per il filtro passa basso"""

        dilation: int
        """DRAWidthHeight.Input.dilation (int): dilation per il kernel del filtro passa basso"""

        kernel: int
        """DRAWidthHeight.Input.kernel (int): dimensione del kernel per la convoluzione del filtro passa basso"""

        device: torch.device = torch.device("cpu")

    def __init__(self, conf: Input) -> None:
        """DRAWidthHeight.__init__: Costruzione dei layer DRA lungo orizzontale e verticale e dei parametri di peso per
        output primario e ramo del residuo
        Args:
            conf (DRAWidthHeight.Input): Configurazione del layer .
        Returns:
            None
        """

        super().__init__()

        self.H_spatial_att = DilatedAttention(
            DilatedAttention.Input(
                inchannels=conf.dim,
                mode=DilatedAttention.SpatialMode.DILATED_RECTAGLE_ATTENTION_HEIGHT,
                kernel_size=conf.kernel,
                dilation=conf.dilation,
                group=conf.group,
            )
        )
        self.W_spatial_att = DilatedAttention(
            DilatedAttention.Input(
                inchannels=conf.dim,
                mode=DilatedAttention.SpatialMode.DILATED_RECTAGLE_ATTENTION_WIDTH,
                kernel_size=conf.kernel,
                dilation=conf.dilation,
                group=conf.group,
            )
        )
        self.gamma = nn.Parameter(torch.zeros(conf.dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(conf.dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DRAWidthHeight.forward: implementa il forward pass per DRA
        Args:
            x (torch.Tensor): Minibatch in input, ndims = 4, con x.shape[1] == conf.dim
        Returns:
            (torch.Tensor) output del layer dato il minibatch, somma pesata tra output sequenza DRAs e residuo (input)
        """

        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class MultiShapeAttention(nn.Module):
    """MultiShapeAttention: torch.nn.Module implementante MSA (originale MultiShapeKernel) tramite l'applicazione di DSA e DRA horz+vert
    Attributes:
        square_att (DilatedAttention): layer DSA
        strip_att (DRAWidthHeight): layer DRA horz+vert
    """

    class Input(NamedTuple):
        """MultiShapeAttention.Input: Parametri del costruttore di MultiShapeAttention
        Attributes:
            dim (int): numero di canali in input attesi
            kernel_size (int): kernel size per la convoluzione con il filtro passa basso in DSA e DRA (default = 3)
            dilation (int): dilation per il kernel passa basso per DSA e DRA (default = 1)
            group (int): numero gruppi per la convoluzione filtro passa basso per DSA e DRA (default = 8)
            device (torch.device): Device in cui i parametri devono risiedere
        """

        dim: int
        """MultiShapeAttention.Input.dim (int): numero di canali in input attesi"""

        kernel_size: int = 3
        """MultiShapeAttention.Input.kernel_size (int): kernel size per la convoluzione con il filtro passa basso in DSA e DRA (default = 3)"""

        dilation: int = 1
        """MultiShapeAttention.Input.dilation (int): dilation per il kernel passa basso per DSA e DRA (default = 1)"""

        group: int = 8
        """MultiShapeAttention.Input.group (int): numero gruppi per la convoluzione filtro passa basso per DSA e DRA (default = 8)"""

        device: torch.device = torch.device("cpu")
        """MultiShapeAttention.Input.device (torch.device): Device in cui i parametri devono risiedere"""

    def __init__(self, conf: Input) -> None:
        """MultiShapeAttention.__init__: Costruzione dei layer DRA e DSA
        Args:
            conf (MultiShapeAttention.Input): Configurazione del layer
        Returns:
            None
        """

        super().__init__()
        self.square_att = DilatedAttention(
            DilatedAttention.Input(
                inchannels=conf.dim,
                mode=DilatedAttention.SpatialMode.DILATED_SQUARE_ATTENTION,
                dilation=conf.dilation,
                group=conf.group,
                kernel_size=conf.kernel_size,
            )
        )
        self.strip_att = DRAWidthHeight(
            DRAWidthHeight.Input(
                dim=conf.dim,
                group=conf.group,
                dilation=conf.dilation,
                kernel=conf.kernel_size,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MultiShapeAttention.forward: implementa il forward pass per MSA
        Args:
            x (torch.Tensor): Minibatch in input, ndims = 4, con x.shape[1] == conf.dim
        Returns:
            (torch.Tensor) output del layer MSA, dato dala somma output DSA e DRA -> Parallelizzabile
        """

        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1 + x2


class MultiScaleModule(nn.Module):
    """MultiScaleModule: nn.Module implementante MSM, dunque prende un input, e gli somma versioni downsampled fatte
    passare per AP, MSA. a ciascuna versione downsampled vendono sommati tutti i mips di livello superiore
    Attributes:
        pools_sizes (list[int]): lista di fattori di downsampling per ciascun mip (fissa a [8, 4, 2])
        pools (nn.ModuleList): lista di layers di average pooling da applicare ad ogni mip del volume di input
        convs (nn.ModuleList): lista di convoluzioni 3x3 da applicare prima di passare il mip al MSA
        dynas (nn.ModuleList): lista di moduli MSA, uno per ogni mip
        relu (nn.GELU): funzione di attivazione applicata soltanto al mip con fattore di downsampling piu basso (/2)
        conv_sum (nn.Conv2d): convoluzione 3x3 finale applicata sulla somma dell'elaborato di tutti i mip
    """

    class Input(NamedTuple):
        """MultiScaleModule.Input: Parametri del costruttore di MultiScaleModule
        Attributes:
            k (int): numero canali in input atteso
            k_out (int): numero canali da dare in output
        """

        k: int
        """MultiScaleModule.Input.k (int): numero canali in input atteso"""

        k_out: int
        """MultiScaleModule.Input.k_out (int): numero canali da dare in output"""

    def __init__(self, conf: Input) -> None:
        """MultiScaleModule.__init__: Costruzione dei layer MSM
        Args:
            conf (MultiScaleModule.Input): Configurazione del layer
        Returns:
            None
        """

        super().__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [7, 9, 11]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(conf.k, conf.k, 3, 1, 1, bias=False))
            dynas.append(
                MultiShapeAttention(
                    MultiShapeAttention.Input(
                        dim=conf.k, kernel_size=3, dilation=dilation[j]
                    )
                )
            )
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(conf.k, conf.k_out, 3, 1, 1, bias=False)

    # -----------------------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_size = x.size()
        resl = x
        y_up = None
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](
                    self.convs[i](self.pools[i](x))
                )  # quindi il conv3x3 e' fatto fuori dal DSA
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(
                resl,
                nnfunc.interpolate(y, x_size[2:], mode="bilinear", align_corners=True),
            )
            if i != len(self.pools_sizes) - 1:
                y_up = nnfunc.interpolate(
                    y, scale_factor=2, mode="bilinear", align_corners=True
                )
        resl = self.relu(resl)  # perche' relu solo all'ultimo?
        resl = self.conv_sum(resl)

        return resl


class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        # forse da spostare sul device giusto
        self.square_att = DilatedAttention(
            DilatedAttention.Input(
                inchannels=dim,
                mode=DilatedAttention.SpatialMode.DILATED_SQUARE_ATTENTION,
                dilation=dilation,
                group=group,
                kernel_size=kernel_size,
            )
        )
        self.strip_att = DRAWidthHeight(
            DRAWidthHeight.Input(
                dim=dim, group=group, dilation=dilation, kernel=kernel_size
            )
        )

    def forward(self, x):
        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1 + x2


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [7, 9, 11]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        y_up = 0
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(
                resl,
                nnfunc.interpolate(y, x_size[2:], mode="bilinear", align_corners=True),
            )
            if i != len(self.pools_sizes) - 1:
                y_up = nnfunc.interpolate(
                    y, scale_factor=2, mode="bilinear", align_corners=True
                )
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl


class ResBlock(nn.Module):
    """ResBlock: nn.Module Implementante meta' del CNNBlock. Il CNNBlock e' composto da due di questi res block, il
    primo senza MSM e il secondo con MSM"""

    class Input(NamedTuple):
        in_channel: int
        out_channel: int
        has_msm: bool = False

    def __init__(self, conf: Input) -> None:
        super().__init__()
        conv1 = BasicConv(
            BasicConv.Input(
                in_channel=conf.in_channel,
                out_channel=conf.out_channel,
                kernel_size=3,
                stride=1,
            )
        )
        msm = MultiScaleModule(
            MultiScaleModule.Input(k=conf.in_channel, k_out=conf.out_channel)
        )
        conv2 = BasicConv(
            BasicConv.Input(
                conf.out_channel, conf.out_channel, kernel_size=3, stride=1, relu=False
            )
        )
        self.main = nn.Sequential(
            OrderedDict(
                [("Conv1", conv1)] + [("MSM", msm)]
                if conf.has_msm
                else [] + [("Conv2", conv2)]
            )
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [
            ResBlock(ResBlock.Input(in_channel=out_channel, out_channel=out_channel))
            for _ in range(num_res - 1)
        ]
        layers.append(
            ResBlock(
                ResBlock.Input(
                    in_channel=out_channel, out_channel=out_channel, has_msm=True
                )
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [
            ResBlock(ResBlock.Input(in_channel=channel, out_channel=channel))
            for _ in range(num_res - 1)
        ]
        layers.append(
            ResBlock(
                ResBlock.Input(in_channel=channel, out_channel=channel, has_msm=True)
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(
                BasicConv.Input(
                    in_channel=3,
                    out_channel=out_plane // 4,
                    kernel_size=3,
                    stride=1,
                    relu=True,
                )
            ),
            BasicConv(
                BasicConv.Input(
                    in_channel=out_plane // 4,
                    out_channel=out_plane // 2,
                    kernel_size=1,
                    stride=1,
                    relu=True,
                )
            ),
            BasicConv(
                BasicConv.Input(
                    in_channel=out_plane // 2,
                    out_channel=out_plane // 2,
                    kernel_size=3,
                    stride=1,
                    relu=True,
                )
            ),
            BasicConv(
                BasicConv.Input(
                    in_channel=out_plane // 2,
                    out_channel=out_plane,
                    kernel_size=1,
                    stride=1,
                    relu=False,
                )
            ),
            nn.InstanceNorm2d(out_plane, affine=True),
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(
            BasicConv.Input(
                in_channel=channel * 2,
                out_channel=channel,
                kernel_size=3,
                stride=1,
                relu=False,
            )
        )

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class ConvIR(nn.Module):
    def __init__(self, num_res=16):
        super(ConvIR, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )

        self.feat_extract = nn.ModuleList(
            [
                BasicConv(
                    BasicConv.Input(
                        in_channel=3,
                        out_channel=base_channel,
                        kernel_size=3,
                        relu=True,
                        stride=1,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel,
                        out_channel=base_channel * 2,
                        kernel_size=3,
                        relu=True,
                        stride=2,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 2,
                        out_channel=base_channel * 4,
                        kernel_size=3,
                        relu=True,
                        stride=2,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 4,
                        out_channel=base_channel * 2,
                        kernel_size=4,
                        relu=True,
                        stride=2,
                        transpose=True,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 2,
                        out_channel=base_channel,
                        kernel_size=4,
                        relu=True,
                        stride=2,
                        transpose=True,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel,
                        out_channel=3,
                        kernel_size=3,
                        relu=False,
                        stride=1,
                    )
                ),
            ]
        )

        self.Decoder = nn.ModuleList(
            [
                DBlock(base_channel * 4, num_res),
                DBlock(base_channel * 2, num_res),
                DBlock(base_channel, num_res),
            ]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 4,
                        out_channel=base_channel * 2,
                        kernel_size=1,
                        relu=True,
                        stride=1,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 2,
                        out_channel=base_channel,
                        kernel_size=1,
                        relu=True,
                        stride=1,
                    )
                ),
            ]
        )

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 4,
                        out_channel=3,
                        kernel_size=3,
                        relu=False,
                        stride=1,
                    )
                ),
                BasicConv(
                    BasicConv.Input(
                        in_channel=base_channel * 2,
                        out_channel=3,
                        kernel_size=3,
                        relu=False,
                        stride=1,
                    )
                ),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = nnfunc.interpolate(x, scale_factor=0.5)
        x_4 = nnfunc.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net():
    return ConvIR()
