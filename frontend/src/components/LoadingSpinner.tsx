export default function LoadingSpinner({ label = "Loading..." }: { label?: string }) {
return (
<div className="flex items-center gap-2 text-sm text-gray-400 p-6">
<div className="w-4 h-4 border-2 border-gray-500 border-t-blue-500 rounded-full animate-spin" />
{label}
</div>


);
}