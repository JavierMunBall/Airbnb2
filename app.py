import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("ml_airbnb")
st.title("Sistema de predicción de precios Upgrade-Hub")

neighbourhood = st.selectbox('Barrio', options=[
    'Villa Soldati',' Villa Riachuelo','Versalles','Villa Lugano','Villa Real','Nueva Pompeya','Parque Avellaneda',
    'Monte Castro','Liniers','Velez Sarsfield','Mataderos','Villa Gral. Mitre','Villa Luro','Villa Santa Rita',
    'Floresta','Agronomia','Villa Pueyrredon','Parque Chas','Parque Chacabuco','Parque Patricios','Boca','Boedo',
    'Villa Del Parque','Villa Devoto','Coghlan','Villa Ortuzar','Flores','San Cristobal','Saavedra','Barracas','Puerto Madero',
    'Villa Urquiza','Constitucion','Chacarita','Caballito','Nuñez','Colegiales','San Telmo','Balvanera','Villa Crespo','Almagro',
    'Monserrat','Retiro','Belgrano','San Nicolas','Recoleta','Palermo'
    ])

property_type = st.selectbox('Tipo de Propiedad', options=[
    'Entire condo', 'Entire rental unit', 'Private room in rental unit', 'Entire loft', 'Entire townhouse',
    'Entire serviced apartment', 'Room in aparthotel', 'Private room in home', 'Private room in condo',
    'Private room in casa particular', 'Entire home', 'Private room in bed and breakfast', 'Entire villa',
    'Shared room in rental unit', 'Shared room in guest suite', 'Camper/RV', 'Shared room in hostel', 'Private room',
    'Private room in guesthouse', 'Private room in hostel', 'Private room in villa', 'Private room in loft',
    'Private room in serviced apartment', 'Room in hostel', 'Private room in townhouse', 'Shared room in home',
    'Private room in guest suite', 'Room in serviced apartment', 'Casa particular', 'Shared room in villa', 'Entire guesthouse',
    'Entire vacation home', 'Room in boutique hotel', 'Shared room in loft', 'Entire place', 'Room in hotel',
    'Shared room in townhouse', 'Tiny home', 'Entire guest suite', 'Entire home/apt', 'Private room in chalet', 'Entire in-law',
    'Room in bed and breakfast', 'Shared room in serviced apartment', 'Entire cabin', 'Private room in religious building', 'Campsite',
    'Private room in tiny home', 'Private room in dome', 'Shared room in condo', 'Entire chalet', 'Earthen home', 'Boat',
    'Shared room in guesthouse', 'Pension', 'Private room in dorm', 'Shared room in boutique hotel',
    'Shared room in bed and breakfast', 'Private room in resort', 'Bus', 'Entire cottage', 'Cave', 'Private room in vacation home',
    'Shared room in casa particular', 'Shared room', 'Shared room in barn', 'Tower', 'Shared room in hotel', 'Treehouse', 'Private room in boat'
    ])

accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room'])
maximum_nights = st.slider('Noches Máximas', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

input_data = pd.DataFrame([[
    neighbourhood, property_type, accommodates, room_type,
    maximum_nights, minimum_nights
]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')