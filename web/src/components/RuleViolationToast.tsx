export function RuleViolationToast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className="rule-violation-toast" role="alert">
      <span>{message}</span>
      <button onClick={onDismiss}>×</button>
    </div>
  );
}
