import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Typography, CircularProgress, Button, Box
} from '@mui/material';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label, Legend } from 'recharts';

function SweepResults({ param, values, fixed, epsilon, rounds, numClients, onBack }) {
  const [results, setResults] = useState([]);
  const [loadingIndex, setLoadingIndex] = useState(0);
  const hasRun = useRef(false);

  useEffect(() => {
    if (hasRun.current) return;
    hasRun.current = true;

    const runSweeps = async () => {
      const output = [];
      for (let i = 0; i < values.length; i++) {
        const value = values[i];
        const config = { epsilon, rounds, numClients };
        config[param] = value;

        setLoadingIndex(i + 1);
        try {
          const response = await axios.post('http://localhost:8000/run', config);
          output.push({
            x: Number(value),
            dp_accuracy: response.data.dp_final_accuracy,
            non_dp_accuracy: response.data.non_dp_final_accuracy,
            noise: response.data.average_noise
          });
        } catch (error) {
          console.error(`Error during run with ${param}=${value}:`, error);
          output.push({ x: value, accuracy: null });
        }
      }

      output.sort((a, b) => a.x - b.x);
      setResults(output);
    };

    runSweeps();
  }, [param, values, epsilon, rounds, numClients]);

  return (
    <Box sx={{ padding: 4, textAlign: 'center' }}>
      <Typography variant="h4" gutterBottom>
        Parameter Evaluation Results
      </Typography>
      <Typography variant="body1" gutterBottom>
        Evaluating accuracy across various <strong>{param}</strong> with values: {values.join(', ')}
      </Typography>
      <Box sx={{ textAlign: 'center' }}>
        {Object.entries(fixed).map(([key, value]) => (
          <Typography key={key} variant="body2">
            {key}: <strong>{value}</strong>
          </Typography>
        ))}
      </Box>

      {results.length < values.length ? (
        <Box sx={{ mt: 4 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ mt: 2 }}>
            Running {param} = {values[loadingIndex - 1]}...
          </Typography>
        </Box>
      ) : (
        <>
          <LineChart width={700} height={400} data={results} margin={{ bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              type="number"
              domain={['dataMin', 'dataMax']}
              ticks={values.map(Number)}
              label={{ value: param, position: 'insideBottom', offset: -5 }}
            />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}>
              <Label angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }}>
                Final Accuracy
              </Label>
            </YAxis>
            <Tooltip formatter={(v) => v != null ? `${(v * 100).toFixed(2)}%` : 'Error'} />
            <Line dataKey="dp_accuracy" stroke="#8884d8" name="With DP Noise" connectNulls />
            <Line dataKey="non_dp_accuracy" stroke="#82ca9d" name="Without DP Noise" connectNulls />
            <Legend verticalAlign="top" align="right" />
          </LineChart>
          <Typography variant="caption" display="block" sx={{ mt: 2 }}>
            Accuracy is reported after final round of training for each parameter setting.
          </Typography>

          <Typography variant="h6" sx={{ mt: 6, mb: 1 }}>
            Average noise magnitude per sweep
          </Typography>
          <LineChart
            width={700}
            height={400}
            data={results}
            margin={{ bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              type="number"
              domain={['dataMin', 'dataMax']}
              ticks={values.map(Number)}
              label={{ value: param, position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              label={{
                value: 'Average Noise',
                angle: -90,
                position: 'insideLeft',
                style: { textAnchor: 'middle' }
              }}
            />
            <Tooltip formatter={(v) => v != null ? v.toFixed(3) : 'Error'} />
            <Line dataKey="noise" stroke="#82ca9d" name="Average Noise" connectNulls />
          </LineChart>

        </>
      )}

      <Button variant="outlined" color="secondary" sx={{ mt: 4 }} onClick={onBack}>
        Back to Settings
      </Button>
    </Box>
  );
}

export default SweepResults;
