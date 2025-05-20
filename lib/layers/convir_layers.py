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
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Blocco convolutivo di base
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

# Blocchi encoder
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Blocchi decoder
class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# ConvS nel paper, prepara le immagini degradate più piccole per il merge
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

# Fa il merge delle immagini degradate man mano
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class ConvIR(nn.Module):
    def __init__(self, num_res=4):
        super(ConvIR, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        # Convoluzioni prima di CNNBlock
        self.feat_extract = nn.ModuleList([
            # Encoder
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            
            # Decoder
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        # Convoluzione dopo skip connection lato decoder
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # Convoluzioni trasposte per generare output intermedi
        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
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
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net(n: int):
    return ConvIR(num_res=n)