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

import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
import logging
from typing import NamedTuple


class BasicConv(nn.Module):
    """BasicConv: Classe implementante un layer convolutivo (sia in downsampling che in transpose) sandwitched da operazioni comuni, ovvero
    - Batch Normalization
    - Nonlinearity di tipo ReLU
    Attributes:
        main: nn.Sequential contenente le operazioni compiute dal layer convolutivo
        device: torch.device in cui gli nn.Parameters del modello risiedono
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

        stride: int | (int, int)
        """BasicConv.Input.stride: Displacement in orizzontale e verticale del kernel nell'input da una locazione spaziale alla successiva."""

        bias: bool = True
        """BasicConv.Input.bias: Booleano indicante se aggiungere il termine di bias in ogni convoluzione o no."""

        norm: bool = False
        """BasicConv.Input.norm: Booleano indicante se fare la batch normalization o meno."""

        relu: bool = True
        """BasicConv.Input.relu: Booleano indicante se applicare nonLinearity o meno."""

        transpose: bool = False
        """BasicConv.Input.transpose: Booleano indicante se applicare convoluzione in downsampling o convoluzione trasposta."""

        device: torch.device = torch.device('cpu')
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
            layers.append(nn.ConvTranspose2d(
                conf.in_channel, conf.out_channel, conf.kernel_size, padding=padding, stride=conf.stride, bias=bias))
        else:
            # aggiungi padding preciso per preservare la dimensione spaziale con stride unitario
            padding = conf.kernel_size // 2
            layers.append(nn.Conv2d(
                conf.in_channel, conf.out_channel, conf.kernel_size, padding=padding, stride=conf.stride, bias=bias))
        if conf.norm:
            layers.append(nn.BatchNorm2d(conf.out_channel))
        if conf.relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)
        self.device = conf.device

        self.to(self.device)
        match self.device.type:
            case 'cuda':
                # da rimuovere
                logging.info('[BasicConv] Created BasicConv layer on the GPU')
            case x if x != 'cpu':
                # da rimuovere
                logging.warning('[BasicConv] Created BasicConv layer on unrecognized device')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward: implementa il forward pass per Basic Conv, prende un minibatch x e ne calcola l'uscita y
        Args:
            x (torch.Tensor): Minibatch in input
        Returns:
            (torch.Tensor) output del layer dato il minibatch
        """

        return self.main.forward(x)


class DynamicFilter(nn.Module):
    """DynamicFilter: nn.Module dedito ad applicare Layer DSA, *Dilated Square Attention*
    """

    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(DynamicFilter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Tanh()

        # inizializza i parametri del self.conv della rete secondo il criterio "He Initialization"
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True) # peso low pass filter
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True) # peso high pass filter
        self.pad = nn.ReflectionPad2d(self.dilation * (kernel_size - 1) // 2)   # calcola padding per preservare size

        self.ap = nn.AdaptiveAvgPool2d((1, 1)) # sti due sono uguali ma con parametri diversi
        self.gap = nn.AdaptiveAvgPool2d(1) # -> AdaptiveAvgPool2d((1,1))(randn(10, 3, 4, 4)).shape = Size([10, 3, 1, 1])

        self.inside_all = nn.Parameter(torch.zeros(inchannels, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity_input = x

        # 1: Low pass frequency extraction
        low_filter = self.ap.forward(x) # (c = 3) prendi il colore medio per ogni x_i in minibatch
        low_filter = self.conv.forward(low_filter) # convoluzione 1x1 per avere 'group x kernel_size^2' canali
        low_filter = self.bn.forward(low_filter) # dopo ogni convoluzione BatchNorm ci sta
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
        x = (nnfunc.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation) #-> xn c*k2 hw
                   .reshape(n, self.group, x_c // self.group, self.kernel_size ** 2, x_h * x_w)) #-> xn g c/g k2 hw
        # Questa operazione ha la duplice funzione di facilitare 'Grouped Operations' e 'Operazioni Patch-Wise' (implementando il concetto di spatial attention)
        # Grouped Operations   = Suddividere i canali in gruppi e processare ciascun gruppo in parallelo
        # Patch-Wise Filtering = Allinea le colonne con la distribuzione spaziale del filtro

        # 3: Prepara anche il low pass filter per fare grouped convolutions
        l_n, low_filter_c, low_filter_h, low_filter_w = low_filter.shape # dalla size filter scritta sopra passo a ...
        if l_n != n or low_filter_h * low_filter_w != 1:
            raise ValueError('io della vita non ho capito un')

        # ricorda che unsqueeze aggiunge una dimensione pari a 1 nell'indice che gli dai. Esempio con x di size 10 3 224 224
        # { 'ini': x.shape, '0': x.unsqueeze(0).shape, '1': x.unsqueeze(1).shape, '2': x.unsqueeze(2).shape, '3': x.unsqueeze(3).shape }
        # { 'ini': [10, 3, 224, 224], '0': [1, 10, 3, 224, 224], '1': [10, 1, 3, 224, 224], '2': [10, 3, 1, 224, 224]), '3': [10, 3, 224, 1, 224] }
        low_filter = (low_filter.reshape( # ... n, ((group x kernel_size^2) / kernel_size^2), kernel_size^2, (1 x 1)
            n, low_filter_c // self.kernel_size ** 2, self.kernel_size ** 2, low_filter_h * low_filter_w)
                                .unsqueeze(2)) # ... n, group, 1, kernel_size^2, 1
        # hai ottenuto una struttura analoga all'input (n, group, channels/group, kernel_size^2, height x width)
        low_filter = self.act.forward(low_filter) # passa filtro su nonlinearita tanh, caratterizzata da smoothness, bounds, simmetria -> regolarizzazione e controllo

        # 4: Applicazione filtro passa basso
        # x = torch.randn(10, 3, 1, 9, 224 * 224)
        # k = torch.randn(10, 3, 1, 9, 1)
        # prod = x * k # nota che puoi fare il prodotto finche k e' broadcastabile (ciascuna dimensione in x multipla della corrispettiva in k)
        # x.shape, k.shape, prod.shape
        # >> (torch.Size([10, 3, 1, 9, 50176]), torch.Size([10, 3, 1, 9, 1]), torch.Size([10, 3, 1, 9, 50176]))
        # il broadcast in questo caso e' fatto su tutti i canali e posizioni spaziali, applicando il filtro passa basso in one shot
        # dopodiche fa la somma su tutti i prodotti per il kernel, completando il prodotto scalare, e poi riassembla tensore con reshape
        # dot_prod = torch.sum(prod, dim=3)
        # dot_prod.shape
        # >> torch.Size([10, 3, 1, 50176])
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, x_c, x_h, x_w)

        # -----------------------------------------------------------------------------------------------------------------
        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap.forward(identity_input)

        out_low = out_low * self.lamb_l[None, :, None, None]

        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation * (kernel - 1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim, 1, 1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None, 1) if H else (1, None)
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = (nnfunc.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation)
             .reshape(n, self.group, c // self.group, self.k, h * w))
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None, :, None, None]
        out_high = identity_input * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class MultiShapeKernel(nn.Module):
    class Input(NamedTuple):
        dim: int
        kernel_size: int = 3
        dilation: int = 1
        group: int = 8

    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = DynamicFilter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):
        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1 + x2


class DeepPoolLayer(nn.Module):
    class Input(NamedTuple):
        k: int
        k_out: int

    def __init__(self, conf: Input):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [7, 9, 11]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(conf.k, conf.k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=conf.k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(conf.k, conf.k_out, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_size = x.size()
        resl = x
        y_up = None
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i].forward(self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i].forward(self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(resl, nnfunc.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes) - 1:
                y_up = nnfunc.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu.forward(resl)
        resl = self.conv_sum.forward(resl)

        return resl


class ResBlock(nn.Module):
    class Input(NamedTuple):
        in_channel: int
        out_channel: int
        filter_: bool = False

    def __init__(self, conf: Input) -> None:
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(BasicConv.Input(
                in_channel=conf.in_channel,
                out_channel=conf.out_channel,
                kernel_size=3,
                stride=1
            )),
            DeepPoolLayer(
                DeepPoolLayer.Input(k=conf.in_channel, k_out=conf.out_channel)) if conf.filter_ else nn.Identity(),
            BasicConv(BasicConv.Input(conf.out_channel, conf.out_channel, kernel_size=3, stride=1, relu=False))
        )

    def forward(self, x):
        return self.main(x) + x
