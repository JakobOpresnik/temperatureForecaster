import { useEffect, useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';
import supabase from './supabaseClient';
import type { Database } from './types/supabase';
import { MapContainer, Marker, Popup, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { LineChart } from '@mui/x-charts';
type WeatherRow = Database['public']['Tables']['weather']['Row'];

type Forecast = {
  station: string;
  predictions: number[];
  actuals: number[];
  timestamps: string[];
};

type StationMarkerProps = {
  station: string;
  latitude: number;
  longitude: number;
};

function App() {
  const [count, setCount] = useState(0);
  const [data, setData] = useState<WeatherRow[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [forecasts, setForecasts] = useState<Forecast[]>([]);

  const getRegisteredModels = async () => {
    try {
      const response: Response = await fetch('http://localhost:8000/');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('DATA: ', data);
      setModels(data);
    } catch (err) {
      console.log(err);
    }
  };

  const getTemperatureForecast = async (station: string) => {
    try {
      const response: Response = await fetch(
        `http://localhost:8000/predict/${station}`
      );
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      console.log('fetched data: ', data);
      const forecast: Forecast = {
        station: station,
        predictions: data.predictions,
        actuals: data.actuals,
        timestamps: data.timestamps,
      };
      setForecasts((prev: Forecast[]) => [...prev, forecast]);
    } catch (err) {
      console.log(err);
    }
  };

  const getData = async () => {
    const { data, error } = await supabase.from('weather').select('*').limit(5);
    if (error) {
      console.error('Supabase connection test failed:', error);
    } else {
      console.log('Supabase connection test succeeded:', data);
      setData(data);
    }
  };

  useEffect(() => {
    getData();
    getRegisteredModels();
  }, []);

  useEffect(() => {
    for (const model of models) {
      getTemperatureForecast(model.split('-')[1]);
    }
  }, [models]);

  console.log('forecasts: ', forecasts);

  function getNextTimestamps(start: string, count: number): string[] {
    console.log('start: ', start);
    const [hours, minutes] = start.split(':').map(Number);
    const startDate = new Date();
    startDate.setHours(hours, minutes, 0, 0);

    const timestamps: string[] = [];

    for (let i = 1; i <= count; i++) {
      const next = new Date(startDate.getTime() + i * 30 * 60 * 1000);
      const hh = next.getHours().toString().padStart(2, '0');
      const mm = next.getMinutes().toString().padStart(2, '0');
      timestamps.push(`${hh}:${mm}`);
    }

    return timestamps;
  }

  const StationMarker = ({
    station,
    latitude,
    longitude,
  }: StationMarkerProps) => {
    const forecast: Forecast | undefined = forecasts.find(
      (forecast: Forecast) => forecast.station === station
    );

    /* const customIcon = new Icon({
      iconUrl: 'assets/location_pin.png',
      iconSize: [45, 45],
    }); */

    const chartData: number[] = [
      ...(forecast?.actuals ?? []),
      ...(forecast?.predictions ?? []),
    ];

    const forecastTimestamps: string[] = getNextTimestamps(
      forecast?.timestamps[17] ?? '',
      6
    );

    console.log('timestamps forecast: ', forecastTimestamps);
    console.log([...(forecast?.timestamps ?? []), ...forecastTimestamps]);

    return (
      <Marker position={[latitude, longitude]}>
        <Popup>
          <h3>{station}</h3>
          {forecast?.predictions?.map(
            (value: number) => `${value.toFixed(1)}°C`
          )}
          {forecast?.timestamps && (
            <LineChart
              series={[
                {
                  data: chartData,
                  area: true,
                },
              ]}
              /* xAxis={[
                {
                  data: [...forecast.timestamps, ...forecastTimestamps],
                },
              ]} */
              yAxis={[
                {
                  min: Math.min(...chartData) * 0.85,
                  max: Math.max(...chartData) * 1.15,
                },
              ]}
              height={300}
              width={350}
            />
          )}
        </Popup>
      </Marker>
    );
  };

  return (
    <>
      {/* <ul>
        <button onClick={() => getTemperatureForecast('Lendava')}>
          GET PREDICTIONS FOR LENDAVA
        </button>
        {forecasts.map((forecast: Forecast, index: number) => (
          <p key={index}>
            {forecast.station}:{' '}
            {forecast.predictions?.map(
              (value: number) => `${value.toFixed(1)}°C `
            )}
          </p>
        ))}
        {models.map((model: string, index: number) => (
          <li key={index}>{model}</li>
        ))}
        {data.map((weather: WeatherRow, index: number) => (
          <li key={index}>
            {weather.Date} {weather.Location} {weather.Temperature}
          </li>
        ))}
      </ul> */}

      <MapContainer
        center={[46.173785, 14.973106]}
        zoom={9}
        style={{ height: '100vh' }}
      >
        <TileLayer url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png' />
        <StationMarker
          station='LENDAVA'
          latitude={46.562634}
          longitude={16.452506}
        />
        <StationMarker
          station='PTUJ'
          latitude={46.418545}
          longitude={15.869091}
        />
        <StationMarker
          station='CELJE'
          latitude={46.238006}
          longitude={15.269033}
        />
        <StationMarker
          station='CRNOMELJ'
          latitude={45.572651}
          longitude={15.190234}
        />
        <StationMarker
          station='BOVEC'
          latitude={46.338401}
          longitude={13.552329}
        />
      </MapContainer>
    </>
  );

  return (
    <>
      <div>
        <a href='https://vite.dev' target='_blank'>
          <img src={viteLogo} className='logo' alt='Vite logo' />
        </a>
        <a href='https://react.dev' target='_blank'>
          <img src={reactLogo} className='logo react' alt='React logo' />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className='card'>
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className='read-the-docs'>
        Click on the Vite and React logos to learn more
      </p>
    </>
  );
}

export default App;
