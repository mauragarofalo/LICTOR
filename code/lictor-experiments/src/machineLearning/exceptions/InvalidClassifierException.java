package machineLearning.exceptions;


import java.lang.Exception;

public class InvalidClassifierException extends Exception {

	private static final long serialVersionUID = -7665581268972110249L;

	public InvalidClassifierException() {
		super("The classifier provided as input does not exist");
	}

	public InvalidClassifierException(String pMessage) {
		super(pMessage);
	}
}