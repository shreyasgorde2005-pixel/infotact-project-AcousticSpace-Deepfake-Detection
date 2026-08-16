import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Upload from "./pages/Upload";
import History from "./pages/History";
import Dashboard from "./pages/Dashboards";
import ModelInfoBadge from "./components/ModelInfoBadge";

export default function App() {
  return (
    <BrowserRouter>
      <nav className="flex flex-wrap items-center gap-4 p-4 border-b border-gray-700">
        <Link to="/" className="font-semibold text-white">AcousticSpace</Link>
        <Link to="/" className="hover:text-blue-400">Upload</Link>
        <Link to="/history" className="hover:text-blue-400">History</Link>
        <Link to="/dashboard" className="hover:text-blue-400">Dashboard</Link>
        <ModelInfoBadge />
      </nav>
      <Routes>
        <Route path="/" element={<Upload />} />
        <Route path="/history" element={<History />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

