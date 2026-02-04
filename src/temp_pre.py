import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pytz

class DigitalTwinModel(nn.Module):
    def __init__(self, lookback_steps=144, forecast_steps=18):
        super(DigitalTwinModel, self).__init__()
        self.lookback_steps = lookback_steps
        self.forecast_steps = forecast_steps # 18 steps = 3 hours
        self.tz = pytz.timezone('Europe/Amsterdam')
        
        self.input_dim = None
        self.target_rooms = [] 
        self.net = None

    def prepare_clean_df(self, df_merged):
        df = df_merged.copy()
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
            df.set_index('Timestamp', inplace=True)
        df = df.tz_convert(self.tz)

        # 1. Filter out all watermeter-related columns entirely
        df = df[[c for c in df.columns if 'watermeter' not in c.lower()]]

        # 2. Identify Target Rooms (Living spaces only)
        # Excludes columns ending in '.set' or containing 'watermeter'
        self.target_rooms = [c for c in df.columns if c.lower().endswith('temperature')]
        
        # 3. Identify All Features (PIR, Setpoints, Temperatures)
        keywords = ['temperature', 'set', 'pir']
        feature_cols = [c for c in df.columns if any(k in c.lower() for k in keywords)]
        df = df[feature_cols]

        # 4. Resampling Logic
        df_resampled = df.resample('10min').agg({
            c: ('max' if 'pir' in c.lower() else 'mean') for c in df.columns
        })
        df_resampled = df_resampled.interpolate(method='linear').ffill().bfill()

        # 5. Time Engineering (Cyclical Encoding)
        df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled.index.hour / 24)
        df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled.index.hour / 24)
        df_resampled['day_sin'] = np.sin(2 * np.pi * df_resampled.index.dayofweek / 7)
        df_resampled['day_cos'] = np.cos(2 * np.pi * df_resampled.index.dayofweek / 7)
        
        return df_resampled

    def dataframe_to_tensor(self, df_processed):
        df = df_processed.copy()
        
        # Normalization: $T_{norm} = \frac{T_{actual} - 10}{35}$
        temp_related = [c for c in df.columns if 'temperature' in c.lower()]
        for col in temp_related:
            df[col] = (df[col] - 10) / 35
            
        pir_cols = [c for c in df.columns if 'pir' in c.lower()]
        for col in pir_cols:
            df[col] = df[col].clip(0, 1)

        # Shift Cyclical Features to [0, 1]
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        for col in time_cols:
            df[col] = (df[col] + 1) / 2

        df = df.clip(0, 1)
        
        if len(df) > self.lookback_steps:
            df = df.iloc[-self.lookback_steps:]
            
        self.input_dim = df.shape[1]
        return torch.tensor(df.values, dtype=torch.float32)

    def init_network(self, model_path=None):
        num_targets = len(self.target_rooms)
        output_dim = num_targets * self.forecast_steps
        
        # Expanded MLP to handle multi-room complexity
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.lookback_steps * self.input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim) 
        )

        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.eval()
            print(f"✅ Weights loaded for {num_targets} rooms.")
        else:
            print(f"⚠️ Random weights initialized for {num_targets} rooms.")

    def forward(self, x):
        return self.net(x)

    def predict_future(self, input_tensor):
        if input_tensor.dim() == 2:
            x = input_tensor.unsqueeze(0)
        else:
            x = input_tensor

        with torch.no_grad():
            raw_out = self.forward(x) 
            room_forecasts = raw_out.view(len(self.target_rooms), self.forecast_steps)
        
        result_dict = {
            "meta": {
                "type": "Multi-Room Prediction (No Watermeter)",
                "horizon": "3 Hours",
                "resolution": "10 min"
            },
            "rooms": {}
        }

        # Denormalize: $T_{actual} = T_{norm} \times 35 + 10$
        for i, room_name in enumerate(self.target_rooms):
            room_data = room_forecasts[i].tolist()
            actual_temps = [round(t * 35 + 10, 2) for t in room_data]
            
            result_dict["rooms"][room_name] = [
                {"offset_min": (j+1)*10, "temp": t} 
                for j, t in enumerate(actual_temps)
            ]
            
        return result_dict