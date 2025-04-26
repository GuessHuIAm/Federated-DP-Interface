function PrevArrow(props) {
    const { onClick } = props;
    return (
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '-10px',
          zIndex: 5,
          fontSize: '50px',
          cursor: 'pointer',
          color: '#1976d2',
          transform: 'translate(0, -50%)',
        }}
        onClick={onClick}
      >
        ◀
      </div>
    );
  }
  
  function NextArrow(props) {
    const { onClick } = props;
    return (
      <div
        style={{
          position: 'absolute',
          top: '50%',
          right: '-10px',
          zIndex: 5,
          fontSize: '50px',
          cursor: 'pointer',
          color: '#1976d2',
          transform: 'translate(0, -50%)',
        }}
        onClick={onClick}
      >
        ▶
      </div>
    );
  }

export { PrevArrow, NextArrow };
  