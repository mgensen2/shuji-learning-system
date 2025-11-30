window.addEventListener('load', () => {
    const canvas = document.getElementById('replayCanvas');
    const ctx = canvas.getContext('2d');
    const fileInput = document.getElementById('csvFileInput');
    const drawBtn = document.getElementById('drawBtn');

    // --- 設定項目 (記録ツールと合わせる) ---
    const GRID_SIZE = 8;
    const COORD_LIMIT = 200;
    
    // キャンバスサイズ (初期値)
    let canvasSize = 600; 
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    // ----------------------------------------
    // イベントリスナー
    // ----------------------------------------
    drawBtn.addEventListener('click', handleFileSelect);

    // ----------------------------------------
    // メイン処理
    // ----------------------------------------
    function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) {
            alert("CSVファイルを選択してください。");
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            
            // CSVの種類を判定してパース
            let data = [];
            if (text.includes("event_type")) {
                console.log("Detected Full Data CSV");
                data = parseFullCSV(text);
            } else if (text.includes("Command")) {
                console.log("Detected Plotter CSV");
                data = parsePlotterCSV(text);
            } else {
                alert("不明なCSV形式です。");
                return;
            }

            if (data.length > 0) {
                drawReplay(data);
            } else {
                alert("有効なデータが見つかりませんでした。");
            }
        };
        reader.readAsText(file);
    }

    // ----------------------------------------
    // CSVパース処理 (Full Data)
    // ----------------------------------------
    function parseFullCSV(text) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim());
        
        const idxType = headers.indexOf('event_type');
        const idxX = headers.indexOf('x');
        const idxY = headers.indexOf('y');
        const idxPress = headers.indexOf('pressure');
        const idxStroke = headers.indexOf('stroke_id');

        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            if (row.length < headers.length) continue;

            data.push({
                type: 'full', // データソース識別用
                event_type: row[idxType].trim(),
                x: parseFloat(row[idxX]),
                y: parseFloat(row[idxY]),
                pressure: idxPress !== -1 ? parseFloat(row[idxPress]) : 1.0,
                stroke_id: idxStroke !== -1 ? parseInt(row[idxStroke]) : 0,
                is_drawing: row[idxType].trim() !== 'up' // up以外は描画候補
            });
        }
        return data;
    }

    // ----------------------------------------
    // CSVパース処理 (Plotter Data)
    // ----------------------------------------
    function parsePlotterCSV(text) {
        const lines = text.trim().split('\n');
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim());
        
        // Command,X,Y,Z,F,Delay_ms,Cell_ID
        const idxCmd = headers.indexOf('Command');
        const idxX = headers.indexOf('X');
        const idxY = headers.indexOf('Y');
        const idxZ = headers.indexOf('Z');
        
        if (idxCmd === -1 || idxX === -1 || idxY === -1) {
            alert("プロッタCSVフォーマットエラー: 必要な列が見つかりません。");
            return [];
        }

        const data = [];
        let strokeCount = 0;

        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            if (row.length < headers.length) continue;

            const cmd = row[idxCmd].trim();
            const x = parseFloat(row[idxX]);
            const y = parseFloat(row[idxY]);
            const z = idxZ !== -1 ? parseFloat(row[idxZ]) : 0;

            // A0: 移動 (ペン上げ) -> 新しいストローク開始
            // A1: 描画 (ペン下げ)
            // D1: 待機 (その場に留まる)

            let isDrawing = false;
            
            if (cmd === 'A0') {
                strokeCount++; // 新しいストロークとみなす
                isDrawing = false; // 移動のみ
            } else if (cmd === 'A1') {
                isDrawing = true;
            } else if (cmd === 'D1') {
                isDrawing = true; // 待機中も描画状態継続とみなす(点になる)
            }

            data.push({
                type: 'plotter',
                command: cmd,
                x: x,
                y: y,
                pressure: z,
                stroke_id: strokeCount,
                is_drawing: isDrawing
            });
        }
        return data;
    }

    // ----------------------------------------
    // 描画処理
    // ----------------------------------------
    function drawReplay(data) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        drawGridAndNumbers();

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // ストロークごとに色を変えるためのパレット
        const colors = ['#000000', '#FF0000', '#0000FF', '#008000', '#FFA500', '#800080'];

        // 座標変換関数
        const toPixel = (val, isX) => {
            if (isX) return ((val / COORD_LIMIT) + 1.0) * canvasSize; // X
            else return (val / -COORD_LIMIT) * canvasSize;            // Y
        };

        let isPenDown = false;
        let lastX = null;
        let lastY = null;

        for (let i = 0; i < data.length; i++) {
            const point = data[i];
            const px = toPixel(point.x, true);
            const py = toPixel(point.y, false);
            
            // 筆圧 (Plotterデータの場合はZ=0-8, Fullデータの場合はpressure=0-8)
            const pressure = point.pressure || 0;
            const lineWidth = Math.max(1, pressure * 1.5);

            // 色決定
            const colorIdx = (point.stroke_id - 1) % colors.length;
            const color = colors[colorIdx >= 0 ? colorIdx : 0];

            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = color;
            ctx.fillStyle = color; // 点描画用

            // --- 描画ロジック ---
            if (point.type === 'full') {
                // Full CSV (event_type ベース)
                if (point.event_type === 'down') {
                    isPenDown = true;
                    lastX = px; lastY = py;
                    // 開始点に丸を描く
                    ctx.beginPath();
                    ctx.arc(px, py, lineWidth / 2, 0, Math.PI * 2);
                    ctx.fill();
                } else if (point.event_type === 'move') {
                    if (isPenDown && lastX !== null) {
                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(px, py);
                        ctx.stroke();
                        lastX = px; lastY = py;
                    }
                } else if (point.event_type === 'up') {
                    isPenDown = false;
                    lastX = null; lastY = null;
                }
            } else {
                // Plotter CSV (Command ベース)
                if (point.command === 'A0') {
                    // 移動 (ペン上げ)
                    isPenDown = false;
                    lastX = px; lastY = py; // 次の始点として記録
                } else if (point.command === 'A1' || point.command === 'D1') {
                    // 描画 or 待機
                    if (lastX !== null) {
                        ctx.beginPath();
                        ctx.moveTo(lastX, lastY);
                        ctx.lineTo(px, py);
                        ctx.stroke();
                        
                        // D1 (待機) の場所には強調表示 (オプション)
                        if (point.command === 'D1') {
                            ctx.beginPath();
                            ctx.arc(px, py, lineWidth * 1.5, 0, Math.PI * 2);
                            ctx.fill();
                        }
                    } else {
                        // 最初の点 (A0なしでいきなりA1が来た場合など)
                        ctx.beginPath();
                        ctx.arc(px, py, lineWidth / 2, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    lastX = px; lastY = py;
                }
            }
        }
    }

    // ----------------------------------------
    // グリッドと番号の描画
    // ----------------------------------------
    function drawGridAndNumbers() {
        const cellSize = canvasSize / GRID_SIZE;

        // グリッド線
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 1; i < GRID_SIZE; i++) {
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, canvas.height);
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(canvas.width, i * cellSize);
        }
        ctx.stroke();

        // 外枠
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, canvas.width, canvas.height);

        // セル番号
        ctx.fillStyle = '#ccc'; 
        ctx.font = '20px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let y = 0; y < GRID_SIZE; y++) {
            for (let x = 0; x < GRID_SIZE; x++) {
                const cellId = (y * GRID_SIZE) + x + 1; 
                const centerX = x * cellSize + (cellSize / 2);
                const centerY = y * cellSize + (cellSize / 2);
                ctx.fillText(cellId, centerX, centerY);
            }
        }
    }
    
    drawGridAndNumbers();
});