# -----------------------------------------------------------
# Enthält die Keras-Callbacks
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

from keras.callbacks import Callback, LambdaCallback, LearningRateScheduler, EarlyStopping, ModelCheckpoint
import os


EPOCH_LEARN_RATES = [
    # (epoch to start, learning rate) tuples
    (3, 0.01), (6, 0.008), (12, 0.004), (16, 0.001)
]

def lr_schedule(epoch, lr):
  """ Methode die für bestimmte Epochen die im Array EPOCH_LEARN_RATES definiert sind
      entsprechend die Lernrate setzt
  """
  # Wenn die aktuelle Epoche kleiner ist als das erste Element der EPOCH_LEARN_RATES liste
  # dann bleibt die Standard-Learnrate, die auch beim Compile des Modells angegeben wurde
  # Sonst wird die Lernrate aus der Liste oben genommen, bis dann die Epoche
  # größer wird als das letzte Element der EPOCH_LEARN_RATES list.
  # In dem Fall wird wieder die Standard-Learnrate, 
  # die beim erstellen des Modells angegeben wurde, verwendet
  if epoch < EPOCH_LEARN_RATES[0][0] or epoch > EPOCH_LEARN_RATES[-1][0]:
    return lr
  for i in range(len(EPOCH_LEARN_RATES)):
    if epoch == EPOCH_LEARN_RATES[i][0]:
      return EPOCH_LEARN_RATES[i][1]
  return lr


class TestCallback(Callback):
    ''' Callback erbt den Keras-Callback und ermöglicht das
        Überschreiben der Methoden, die nach jedem "Checkpoint" während
        des Trainigs ausgeführt werden (z.B. on_epoch_begin, on_epoch_end, on_batch_begin usw.)
    '''

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.history = { 'epoch': [], 'test_loss': [], 'test_accuracy': []}

    def on_epoch_end(self, epoch, logs={}):
        ''' Wird nach jeder Epoche aufgerufen.'''
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0, batch_size=self.batch_size)
        self.history['epoch'].append(epoch)
        self.history['test_loss'].append(loss)
        self.history['test_accuracy'].append(acc)
        print(f'\t\t\t\t\t\t\t\t\t\t\t\t test_loss: {loss}, test_accuracy: {acc}\n')

# Diese Callbacks werden zum abspeichern der Modelle bzw. der aktuellen Modellgewichtungen verwendet
# Pfad und Ordner in dem das Modell abgespeichert werden soll
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Callback, dass nach jeder Epoche (period=1) die Modellgewichtungen abspeichert
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_best_only=True,
    save_weights_only=True,
    period=1)