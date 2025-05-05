import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Typography, CircularProgress, Button, Box
} from '@mui/material';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label } from 'recharts';

function SweepResults({ param, values, fixed, epsilon, clip, mechanism, rounds, numClients, onBack }) {
  const [results, setResults] = useState([]);
  const [loadingIndex, setLoadingIndex] = useState(0);
  const hasRun = useRef(false);  // 🚫 prevent duplicate calls

  useEffect(() => {
    if (hasRun.current) return;
    hasRun.current = true;

    const runSweeps = async () => {
        const output = [];
        for (let i = 0; i < values.length; i++) {
          const value = values[i];
      
          const config = {
            epsilon,
            clip,
            mechanism,
            rounds,
            numClients,
          };
          config[param] = value;
      
          setLoadingIndex(i + 1);
      
          try {
            const response = await axios.post('http://localhost:8000/run_once', config);
            output.push({ x: value, accuracy: response.data.final_accuracy });
          } catch (error) {
            console.error(`Error during run with ${param}=${value}:`, error);
            output.push({ x: value, accuracy: null });
          }
        }
      
        // 🔁 Sort x values ascending for better chart display
        output.sort((a, b) => a.x - b.x);
      
        setResults(output);
      };
      
    runSweeps();
  }, [param, values, epsilon, clip, mechanism, rounds, numClients]);

  return (
    <Box sx={{ padding: 4, textAlign: 'center' }}>
      <Typography variant="h4" gutterBottom>
        Parameter Evaluation Results
      </Typography>
      <Typography variant="body1" gutterBottom>
        Evaluating accuracy across various <strong>{param}</strong> with values: {values.join(', ')}
      </Typography>
      <Box sx={{ textAlign: 'center'}}>
        {/* <Typography variant="h6">Fixed Parameters:</Typography> */}
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
          <LineChart width={700} height={400} data={results}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" label={{ value: param, position: 'insideBottom', offset: -5 }} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}>
              <Label angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }}>
                Final Accuracy
              </Label>
            </YAxis>
            <Tooltip formatter={(v) => v != null ? `${(v * 100).toFixed(2)}%` : 'Error'} />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#8884d8"
              name="Final Accuracy"
              connectNulls
            />
          </LineChart>
          <Typography variant="caption" display="block" sx={{ mt: 2 }}>
            Accuracy is reported after final round of training for each parameter setting.
          </Typography>
        </>
      )}

      <Button variant="outlined" color="secondary" sx={{ mt: 4 }} onClick={onBack}>
        Back to Settings
      </Button>
    </Box>
  );
}

export default SweepResults;
