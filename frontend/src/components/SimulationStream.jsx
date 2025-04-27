import { useEffect, useState } from 'react';
import { Button, Typography, CircularProgress, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import TrainingInfoGraphic from './TrainingInfoGraphic';

function SimulationStream({ epsilon, clip, numClients, mechanism, rounds, onBack }) {
  const [globalAccuracy, setGlobalAccuracy] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const [roundDurations, setRoundDurations] = useState([]);
  const [currentRoundStart, setCurrentRoundStart] = useState(Date.now());
  const [totalTrainingTime, setTotalTrainingTime] = useState(0);
  const [eventSource, setEventSource] = useState(null);
  const [currentRoundElapsed, setCurrentRoundElapsed] = useState(0);
  const [clientAccuracy, setClientAccuracy] = useState({});
  const [clientColors, setClientColors] = useState({});
  const [visibleClients, setVisibleClients] = useState({});


  useEffect(() => {
    const source = new EventSource(
      `http://localhost:8000/stream_training?epsilon=${epsilon}&clip=${clip}&num_clients=${numClients}&mechanism=${mechanism}&rounds=${rounds}`
    );
    setEventSource(source);

    source.onmessage = (event) => {
      const data = JSON.parse(event.data);

      const acc = data.global_accuracy.map((a, i) => ({ round: i, accuracy: a }));
      const nextRound = acc.length;
      const newClientAccuracy = {};
      const newClientColors = {};
      const newVisibleClients = {};

      data.client_accuracy.forEach((accHistory, clientIdx) => {
        newClientAccuracy[clientIdx] = accHistory.map((accuracy, round) => ({
          round,
          accuracy
        }));

        newClientColors[clientIdx] = "#" + Math.floor(Math.random()*16777215).toString(16);
        newVisibleClients[clientIdx] = true; // all clients visible initially
      });

      setGlobalAccuracy(acc);
      setClientAccuracy(newClientAccuracy);
      setClientColors(newClientColors);
      setVisibleClients(newVisibleClients);
      setCurrentRound(nextRound);
      setRoundDurations(prev => [...prev, data.round_duration]);
      setTotalTrainingTime(data.total_training_time);
      setCurrentRoundStart(Date.now());
    };

    source.onerror = (e) => {
      console.error("Error in EventSource:", e);
      source.close();
    };

    return () => source.close();
  }, [epsilon, clip, numClients, mechanism, rounds]);
  
  useEffect(() => {
    if (currentRound > rounds) {
      return;
    }

    const id = setInterval(() => {
      setCurrentRoundElapsed((Date.now() - currentRoundStart) / 1000);
    }, 100); // update every 0.1 seconds
  
    return () => clearInterval(id); // cleanup when component unmounts
  }, [currentRoundStart, currentRound, rounds]);

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
    const secString = (secs === 1) ? 'second' : 'seconds';
    const minString = (minutes === 1) ? 'minute' : 'minutes';
    if (minutes > 0) {
      return `${minutes} ${minString}, ${secs} ${secString}`;
    }
    return `${secs} ${secString}`;
  }

  return (
    <div style={{ 
      padding: 20, 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      textAlign: 'center' 
    }}>
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

        {currentRound <= rounds ? (
          <Box sx={{ mt: 1, display: 'inline-flex', alignItems: 'center' }}>
            <CircularProgress size={14} sx={{ mr: 1 }} />
            <Typography variant="body2" component="span">
              Current Round Time: {currentRoundElapsed.toFixed(1)} seconds
            </Typography>
          </Box>
        ) : (
          <Typography variant="body2" sx={{ mt: 1 }}>
            Training complete in {formatDuration(totalTrainingTime)}.
          </Typography>
        )}
      </div>

      <Typography variant="h6" sx={{ mt: 4 }}>Global Training Accuracy</Typography>

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

      <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>Client Training Accuracy</Typography>

      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', gap: 4, mb: 4 }}>
        {/* Left side: the chart */}
        <LineChart width={800} height={500} margin={{ bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
          <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
          <Tooltip
            formatter={(value, name) => [`${(value * 100).toFixed(2)}%`, name]}
            labelFormatter={(label) => label === 0 ? 'Initial Evaluation' : `Round ${label}`}
          />

          {Object.entries(clientAccuracy).map(([clientId, data], idx) => (
            visibleClients[clientId] && (
              <Line
                key={clientId}
                data={data}
                type="monotone"
                dataKey="accuracy"
                name={`Client ${clientId}`}
                stroke={clientColors[clientId]}
                strokeDasharray={idx % 2 === 0 ? "5 5" : "3 3"}
                dot={false}
              />
            )
          ))}
        </LineChart>

        {/* Right side: the checkboxes */}
        <Box
          sx={{
            maxHeight: 500,
            overflowY: 'auto',
            border: '1px solid #ccc',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
            width: 200, // narrow width for the checkbox list
          }}
        >
          {Object.keys(clientAccuracy).map((clientId) => (
            <Box key={clientId} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <input
                type="checkbox"
                checked={visibleClients[clientId]}
                onChange={() =>
                  setVisibleClients(prev => ({
                    ...prev,
                    [clientId]: !prev[clientId]
                  }))
                }
              />
              <Typography sx={{ ml: 1 }} style={{ color: clientColors[clientId], fontSize: '0.8rem' }}>
                Client {clientId}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>

      <Typography variant="h6" sx={{ mt: 4 }}>Round Duration</Typography>

      <LineChart
        width={600}
        height={300}
        data={roundDurations.map((duration, index) => ({ round: index, duration }))}
        margin={{ bottom: 40, top: 20 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
        <YAxis label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }} />
        <Tooltip
          formatter={(value) => [`${value.toFixed(2)}s`, 'Duration']}
          labelFormatter={(label) => label === 0 ? 'Initial Evaluation' : `Round ${label}`}
        />
        <Legend verticalAlign="top" align="right" />
        <Line type="monotone" dataKey="duration" stroke="#82ca9d" activeDot={{ r: 8 }} />
      </LineChart>

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