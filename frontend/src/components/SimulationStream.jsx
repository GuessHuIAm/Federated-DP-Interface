import { useEffect, useState } from 'react';
import { Button, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import TrainingInfoGraphic from './TrainingInfoGraphic';

function SimulationStream({ epsilon, clip, numClients, mechanism, rounds, onBack }) {
  const [globalAccuracy, setGlobalAccuracy] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const [eventSource, setEventSource] = useState(null);

  useEffect(() => {
    const source = new EventSource(
      `http://localhost:8000/stream_training?epsilon=${epsilon}&clip=${clip}&num_clients=${numClients}&mechanism=${mechanism}&rounds=${rounds}`
    );
    setEventSource(source);

    source.onmessage = (event) => {
      const parsedData = JSON.parse(event.data);
      console.log('New training data:', parsedData);

      const updatedGlobalAccuracy = parsedData.global_accuracy.map((acc, index) => ({
        round: index,
        accuracy: acc
      }));

      setGlobalAccuracy(updatedGlobalAccuracy);

      // Instead of trusting round_num, calculate based on globalAccuracy array
      setCurrentRound(updatedGlobalAccuracy.length - 1);
    };

    source.onerror = (err) => {
      console.error('Streaming error:', err);
      source.close();
    };

    return () => {
      if (source) {
        source.close();
      }
    };
  }, [epsilon, clip, numClients, mechanism, rounds]);

  const handleBack = () => {
    if (eventSource) {
      eventSource.close();
    }
    onBack();
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <Typography variant="h5" gutterBottom>
        Training in Progress
      </Typography>

      <TrainingInfoGraphic />

      <div style={{ marginBottom: '20px' }}>
        <Typography variant="body1">Privacy parameter ε: {epsilon}</Typography>
        <Typography variant="body1">Clipping Norm: {clip}</Typography>
        <Typography variant="body1">Number of Clients: {numClients}</Typography>
        <Typography variant="body1">DP Mechanism: {mechanism}</Typography>
        <Typography variant="body1">Rounds: {rounds}</Typography>
        <Typography variant="body2" style={{ marginTop: '10px', fontStyle: 'italic' }}>
          {currentRound === 0
            ? 'Initial Evaluation (Round 0)'
            : `Training Round ${currentRound} / ${rounds}`}
        </Typography>
      </div>

      <LineChart width={600} height={300} data={globalAccuracy}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="round" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="accuracy" stroke="#8884d8" activeDot={{ r: 8 }} />
      </LineChart>

      <Button
        variant="contained"
        color="secondary"
        style={{ marginTop: '30px' }}
        onClick={handleBack}
      >
        Stop and Return to Settings
      </Button>
    </div>
  );
}

export default SimulationStream;