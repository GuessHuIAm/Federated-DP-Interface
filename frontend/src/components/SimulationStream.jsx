import { useEffect, useState, useRef } from 'react';
import { Button, Typography, CircularProgress, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import TrainingInfoGraphic from './TrainingInfoGraphic';

function SimulationStream({ epsilon, clip, numClients, mechanism, rounds, onBack }) {
  const [globalAccuracy, setGlobalAccuracy] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const [roundDurations, setRoundDurations] = useState([]);
  const [currentRoundTime, setCurrentRoundTime] = useState(0);
  const roundStartRef = useRef(null);
  const [eventSource, setEventSource] = useState(null);

  useEffect(() => {
    const source = new EventSource(
      `http://localhost:8000/stream_training?epsilon=${epsilon}&clip=${clip}&num_clients=${numClients}&mechanism=${mechanism}&rounds=${rounds}`
    );
    setEventSource(source);

    roundStartRef.current = Date.now();
    setCurrentRoundTime(0);

    source.onmessage = (event) => {
      const now = Date.now();

      if (roundStartRef.current) {
        setRoundDurations(prev => [
          ...prev,
          (now - roundStartRef.current) / 1000
        ]);
      }

      const data = JSON.parse(event.data);
      const acc = data.global_accuracy.map((a, i) => ({ round: i, accuracy: a }));
      setGlobalAccuracy(acc);
      setCurrentRound(acc.length - 1);

      roundStartRef.current = now;
      setCurrentRoundTime(0);
    };

    source.onerror = (e) => {
      console.error('stream error', e);
      source.close();
    };

    return () => source.close();
  }, [epsilon, clip, numClients, mechanism, rounds]);

  useEffect(() => {
    const id = setInterval(() => {
      if (roundStartRef.current) {
        setCurrentRoundTime(Math.floor((Date.now() - roundStartRef.current) / 1000));
      }
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const handleBack = () => {
    eventSource?.close();
    onBack();
  };

  const roundLabel = () => {
    if (currentRound === 0) return 'Initial Evaluation (Round 0)';
    if (currentRound <= rounds) return `Training Round ${currentRound} / ${rounds}`;
    return 'Training complete';
  };

  function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs < 10 ? '0' : ''}${secs}s`;
  }

  return (
    <div style={{ padding: 20, textAlign: 'center' }}>
      <Typography variant="h5" gutterBottom>Training in Progress</Typography>

      <TrainingInfoGraphic />

      <div style={{ marginBottom: 20 }}>
        <Typography variant="body1">Privacy parameter ε: {epsilon}</Typography>
        <Typography variant="body1">Clipping Norm: {clip}</Typography>
        <Typography variant="body1">Number of Clients: {numClients}</Typography>
        <Typography variant="body1">DP Mechanism: {mechanism}</Typography>
        <Typography variant="body1">Rounds: {rounds}</Typography>

        <Typography variant="h6" sx={{ mt: 1, fontStyle: 'italic' }}>
          {roundLabel()}
        </Typography>

        {currentRound <= rounds && (
          <Box sx={{ mt: 1, display: 'inline-flex', alignItems: 'center' }}>
            <CircularProgress size={14} sx={{ mr: 1 }} />
            <Typography variant="body2" component="span">
              Current Round Time: {formatDuration(currentRoundTime)}
            </Typography>
          </Box>
        )}
      </div>

      <LineChart width={600} height={300} data={globalAccuracy} margin={{ bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
        <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
        <Tooltip
          formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
          labelFormatter={(label) => label === 0 ? 'Initial Evaluation' : `Round ${label}`}
        />
        <Legend verticalAlign="top" align="right" />
        <Line type="monotone" dataKey="accuracy" stroke="#8884d8" activeDot={{ r: 8 }} />
      </LineChart>

      <div style={{ marginTop: 20, textAlign: 'left' }}>
        <Typography variant="h6">Round Durations (seconds):</Typography>
        <ul>
          {roundDurations.map((d, i) => (
            <li key={i}>
              Round {i} : {formatDuration(d)}
            </li>
          ))}
        </ul>
      </div>

      <Button
        variant="contained"
        color="secondary"
        sx={{ mt: 3 }}
        onClick={handleBack}
      >
        Stop and Return to Settings
      </Button>
    </div>
  );
}

export default SimulationStream;