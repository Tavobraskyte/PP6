# Boston Housing Kainų Prognozavimas su TensorFlow/Keras
projektinis darbas Nr. 6

Boston Housing Kainų Prognozavimas su TensorFlow/Keras
Apžvalga
Šiame projekte pateikiamas giluminio mokymosi modelio, sukurtas naudojant TensorFlow ir Keras bibliotekas, taikymas Boston Housing duomenų rinkiniui. Tikslas buvo prognozuoti nekilnojamojo turto kainas pagal įvairius veiksnius, tokius kaip kambarių skaičius, mokesčių rodikliai ir kt.

Projekte buvo atlikti modelio patobulinimai, įskaitant BatchNormalization, Dropout, hiperparametrų derinimą ir įvairias optimizavimo strategijas, kad būtų pasiektas geresnis tikslumas ir sumažintas klaidų lygis. Be to, buvo pridėtos vizualizacijos, kad būtų lengviau stebėti modelio treniravimą.

Tikslai
Duomenų Apdorojimas: Naudojant Boston Housing duomenų rinkinį, duomenys buvo normalizuoti ir padalyti į treniravimo bei testavimo grupes.
Modelio Kūryba ir Treniruotė: Sukurtas gilus neuroninis tinklas su keliomis paslėptomis sluoksniais, naudojant aktyvavimo funkcijas ir optimizavimo algoritmus.
Patobulinimai ir Vizualizacijos: Į modelį buvo įtraukti įvairūs patobulinimai, tokie kaip BatchNormalization, Dropout, ir Adam optimizatorius. Taip pat buvo sukurtos vizualizacijos, kurios rodo modelio veikimo rodiklius per epochas.
Hyperparametrų Tuningas: Naudojant tinkamą hiperparametrų paiešką, buvo pasirinkti geriausi parametrai modelio optimizavimui.
Palyginimas ir Išvados: Įvertintas modelio tikslumas ir atlikti palyginimai tarp skirtingų modelių versijų.
Kodo Struktūra
1. Duomenų Apdorojimas
Duomenys buvo normalizuoti, kad būtų pašalinti bet kokie skirtumai tarp bruožų skalės:

python
Kopijuoti
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
2. Modelio Kūryba
Modelis buvo sukurtas naudojant Keras biblioteką su dviem pilnai sujungtais sluoksniais ir ELU aktyvavimo funkcija. Pirmojo ir antrojo sluoksnių pabaigoje buvo įtrauktas Dropout, kad būtų išvengta per didelės tinklo pritaikymo.

python
Kopijuoti
def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(train_data.shape[1],)))
    model.add(layers.Dense(32))
    model.add(Activation(lambda x: elu(x, alpha=0.1)))
    model.add(Dropout(0.4))

    model.add(layers.Dense(32))
    model.add(Activation(lambda x: elu(x, alpha=0.1)))
    model.add(Dropout(0.4))

    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
3. K-Fold Cross-Validation
Modelis buvo vertinamas naudojant k-fold cross-validation metodą, kad būtų gautas stabilus modelio veikimo įvertinimas. Naudojama 4 dalių validacija.

python
Kopijuoti
k = 4
num_val_samples = len(train_data) // k
num_epochs = 65
bs = 2
all_scores = []
r2_scores = []

for i in range(k):
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=bs, verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    preds = model.predict(val_data)
    r2_scores.append(r2_score(val_targets, preds))
    all_scores.append(val_mae)

print(np.mean(all_scores))
print(np.mean(r2_scores))
4. Vizualizacijos
Modelio treniravimo proceso metu buvo sukurti vizualizacijos, kad būtų galima stebėti klaidų pokyčius ir modelio tikslumą. Šie grafikai padėjo įvertinti modelio progresą per epochos:

python
Kopijuoti
plt.figure(figsize=(12, 6))

# Plot validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), val_loss_mean, label="Validation Loss")
plt.title('Validation Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot validation MAE
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_mae_mean, label="Validation MAE", color='orange')
plt.title('Validation MAE vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
5. Hyperparametrų Tuningas
Įvykdžius hiperparametrų paiešką naudojant tinkamus parametrus (dropout, batch_size, learning_rate), buvo pasirinktas geriausias modelis su optimizuotais parametrais.

python
Kopijuoti
best_params = {
    'dropout_rate': 0.4,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_units': 32
}
6. Galutiniai Rezultatai
Po galutinio modelio apmokymo naudojant visus duomenis, buvo atliktas testavimas su testavimo duomenų rinkiniu ir gauti rezultatai.

python
Kopijuoti
final_model.fit(train_data, train_targets, epochs=65, batch_size=32, verbose=1)
test_loss, test_mae = final_model.evaluate(test_data, test_targets)
Patobulinimai ir Pakeitimai
Vizualizacijos: Naudoti grafikai, kurie rodo validacijos nuostolius ir MAE per epochas, bei regresijos linijos palyginimai tarp tikrų ir prognozuotų reikšmių.
Modelio Tuningas: Naudoti optimizavimo algoritmus Adam ir RMSprop, BatchNormalization ir Dropout pagerinimui.
Hiperparametrų Paieška: Išbandytos įvairios hiperparametrų kombinacijos, siekiant pasirinkti geriausią modelį.
Išvados
Modelio veikimo rodikliai buvo pagerinti naudojant Dropout, BatchNormalization ir Adam optimizavimo algoritmus.
Vizualizacijos padėjo geriau suprasti modelio treniravimo progresą ir jo tikslumą.
Gauti geresni rezultatai palyginus su pradine modelio versija.
Reikalingos Bibliotekos
Norėdami paleisti šį projektą, turite įdiegti šias bibliotekas:

bash
Kopijuoti
pip install tensorflow keras scikit-learn matplotlib
Projekto Pradžia
Duomenų Apdorojimas: Naudokite Boston Housing duomenų rinkinį ir normalizuokite duomenis.
Modelio Treniravimas: Sukurkite modelį ir treniruokite jį naudojant K-fold cross-validation.
Vizualizacijos: Sukurkite vizualizacijas modelio treniravimo rezultatams stebėti.
Hiperparametrų Tuningas: Atlikite hiperparametrų paiešką, kad pasiektumėte geriausią modelio tikslumą.
Testavimas ir Palyginimas: Atlikite modelio testavimą ir įvertinkite jo veikimą naudojant testavimo duomenų rinkinį.
